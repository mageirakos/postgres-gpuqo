/*------------------------------------------------------------------------
 *
 * gpuqo_main.c
 *	  solution to the query optimization problem
 *	  using GPU acceleration
 *
 * src/backend/optimizer/gpuqo/gpuqo_main.c
 *
 *-------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "postgres.h"
#include "miscadmin.h"

#include "optimizer/optimizer.h"
#include "optimizer/cost.h"
#include "optimizer/gpuqo.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"

#define GPUQO_INFO

#ifdef OPTIMIZER_DEBUG
#ifndef GPUQO_INFO
#define GPUQO_INFO
#endif
#endif

int gpuqo_algorithm;

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root);
EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels);
void printQueryTree(QueryTree* qt, int indent);
void printEdges(GpuqoPlannerInfo* info);
RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int n_rels, List *initial_rels);
void fillSelectivityInformation(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfo* info, int n_rels);

/*
 * gpuqo_check_can_run
 *	  checks that there are no join order contraints.
 */
bool 
gpuqo_check_can_run(PlannerInfo* root)
{
    if (list_length(root->join_info_list) != 0)
        return false;

    if (list_length(root->placeholder_list) != 0)
        return false;

    return true;
}

void printQueryTree(QueryTree* qt, int indent){
    int i;

    /* since this function recurses, it could be driven to stack overflow */
	check_stack_depth();

    if (qt == NULL)
        return;

    for (i = 0; i<indent; i++)
        printf(" ");
    printf("%lu (rows=%.0f, cost=%.2f)\n", qt->id->words[0], qt->rows, qt->cost);

    printQueryTree(qt->left, indent + 2);
    printQueryTree(qt->right, indent + 2);
}

void printEdges(GpuqoPlannerInfo* info){
    printf("\nEdges:\n");
    for (int i = 0; i < info->n_rels; i++){
        RelationID edges = info->edge_table[i];
        int idx = -1;
        printf("%d:", i+1);
        while ((idx = bms_next_member(edges, idx)) >= 0){
            EqClassInfo *ec = info->eq_classes;
            
            printf(" %d ", idx);

            if (ec != NULL){
                printf("(");
                while (ec != NULL){
                    if (bms_is_member(i+1, ec->relids) && bms_is_member(idx, ec->relids)){
                        int size = bms_num_members(ec->relids);
                        int idx_l = bms_member_index(ec->relids, i+1);
                        int idx_r = bms_member_index(ec->relids, idx);
                        int idx_sel = eqClassIndex(idx_l, idx_r, size);

                        printf("%lu: %e, ", ec->relids->words[0], ec->sels[idx_sel]);
                    }
                    ec = ec->next;
                }
                printf(");");
            }
        }
        printf("\n");
    }
}

RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int n_rels, List *initial_rels){
    RelOptInfo* left_rel = NULL;
    RelOptInfo* right_rel = NULL;
    RelOptInfo* this_rel = NULL;
    ListCell* lc;

    /* since this function recurses, it could be driven to stack overflow */
	check_stack_depth();

    // baserel
    if (qt->left == 0 || qt->right == 0){
        foreach(lc, initial_rels){
            RelOptInfo* base_rel = (RelOptInfo*) lfirst(lc);
            if (bms_equal(base_rel->relids, qt->id)){
                this_rel = base_rel;
                break;
            }
        }
    } else { // joinrel
        left_rel = queryTree2Plan(qt->left, level-1, root, n_rels, initial_rels);
        right_rel = queryTree2Plan(qt->right, level-1, root, n_rels,initial_rels);
        this_rel = make_join_rel(root, left_rel, right_rel);
        if (this_rel){
            /* Create paths for partitionwise joins. */
            generate_partitionwise_join_paths(root, this_rel);

            /*
                * Except for the topmost scan/join rel, consider gathering
                * partial paths.  We'll do the same for the topmost scan/join
                * rel once we know the final targetlist (see
                * grouping_planner).
                */
            if (level != n_rels)
                generate_gather_paths(root, this_rel, false);

            /* Find and save the cheapest paths for this joinrel */
            set_cheapest(this_rel);
        } 
    }

    if (this_rel == NULL) {
        printf("WARNING: Found NULL RelOptInfo*\n");
    }

    // clean-up the query tree
    pfree(qt);

    return this_rel;
}

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root){
    BaseRelation baserel;
    
    baserel.rows = rel->rows;
    baserel.tuples = rel->tuples;
    baserel.id = bms_copy(rel->relids);
    baserel.fk_selecs = NULL;

    return baserel;
}

EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels){
    ListCell* lc;

    EdgeMask* edge_table = palloc0(sizeof(EdgeMask)*n_rels);
    foreach(lc, root->eq_classes){
        EquivalenceClass *ec = (EquivalenceClass *) lfirst(lc);

        if (list_length(ec->ec_members) <= 1)
			continue;

        for (int i=1; i <= n_rels; i++){
            if (bms_is_member(i, ec->ec_relids)){
                edge_table[i-1] = bms_add_members(edge_table[i-1], ec->ec_relids);
                edge_table[i-1] = bms_del_member(edge_table[i-1], i);
            }
        }
    }

    return edge_table;
}

void fillSelectivityInformation(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfo* info, int n_rels){
    ListCell* lc_inner;
    ListCell* lc_outer;
    ListCell* lc_inner_path;
    ListCell* lc_restrictinfo;
    int i, j;

    info->indexed_edge_table = (EdgeMask*) palloc0(n_rels * sizeof(EdgeMask));

    i = 0;
    foreach(lc_outer, initial_rels){
        RelOptInfo* rel_outer = (RelOptInfo*) lfirst(lc_outer);
        j = 0;
        foreach(lc_inner, initial_rels){
            RelOptInfo* rel_inner = (RelOptInfo*) lfirst(lc_inner);
            
            if (bms_overlap(info->edge_table[i], rel_inner->relids)){
                List *restrictlist = NULL;
                SpecialJoinInfo sjinfo;
                RelOptInfo *join_rel;
                float fk_sel;

                FKSelecInfo* fk = info->base_rels[i].fk_selecs;
                
                sjinfo.type = T_SpecialJoinInfo;
                sjinfo.jointype = JOIN_INNER;
                /* we don't bother trying to make the remaining fields valid */
                sjinfo.lhs_strict = false;
                sjinfo.delay_upper_joins = false;
                sjinfo.semi_can_btree = false;
                sjinfo.semi_can_hash = false;
                sjinfo.semi_operators = NIL;
                sjinfo.semi_rhs_exprs = NIL;
                sjinfo.min_lefthand = rel_outer->relids;
                sjinfo.min_righthand = rel_inner->relids;
                sjinfo.syn_lefthand = rel_outer->relids;
                sjinfo.syn_righthand = rel_inner->relids;

                join_rel = build_join_rel(root,
                    bms_union(rel_outer->relids, 
                                rel_inner->relids),
                    rel_outer,
                    rel_inner,
                    &sjinfo,
                    NULL
                ); 
                
                // I just care about restrictlist
                // but I need to recompute it since it may have been cleaned
                // out by fk selectivities
                restrictlist = build_joinrel_restrictlist(root, join_rel,
                    rel_outer, rel_inner
                );
                
                // recompute selectivities (they will be cached inside the 
                // RestrictInfo), if necessary
                clauselist_selectivity(root,
										restrictlist,
										0,
										sjinfo.jointype,
										&sjinfo);

                foreach(lc_restrictinfo, restrictlist){
                    int size, idx_l, idx_r, idx;
                    RestrictInfo *rinfo= (RestrictInfo*) lfirst(lc_restrictinfo);
                    EqClassInfo* ec = info->eq_classes;

                    if (rinfo->norm_selec == -1)
                        continue;


                    while (ec != NULL){
                        if (bms_equal(ec->relids, rinfo->parent_ec->ec_relids))
                            break;
                        ec = ec->next;

                    };

                    if (ec == NULL){
                        size_t n_sels;
                        
                        ec = (EqClassInfo*) palloc(sizeof(EqClassInfo));
                        ec->next = info->eq_classes;
                        info->eq_classes = ec;
                        ec->relids = bms_copy(rinfo->parent_ec->ec_relids);
                        size = bms_num_members(ec->relids);
                        n_sels = eqClassNSels(size);
                        ec->sels = (float*) palloc(sizeof(float)*n_sels);
                        info->n_eq_classes++;
                        info->n_eq_class_sels += n_sels;
                    } else {
                        size = bms_num_members(ec->relids);
                    }

                    idx_l = bms_member_index(ec->relids, i+1);
                    idx_r = bms_member_index(ec->relids, j+1);
                    if (idx_l < idx_r){ // prevent duplicates
                        idx = eqClassIndex(idx_l, idx_r, size);
                        ec->sels[idx] = rinfo->norm_selec;
                    }
                }

                foreach(lc_inner_path, rel_inner->pathlist){
                    Path* path = (Path*) lfirst(lc_inner_path);
                    if (path->pathtype == T_IndexScan){
                        if (bms_num_members(PATH_REQ_OUTER(path)) == 1 
                                && bms_overlap(PATH_REQ_OUTER(path),
                                    rel_outer->relids
                                )){
                            info->indexed_edge_table[j] = bms_add_member(info->indexed_edge_table[j], i+1);
                            break;
                        }
                    }
                }

                fk_sel = get_foreign_key_join_selectivity(root,
											   rel_outer->relids,
											   rel_inner->relids,
											   &sjinfo,
											   &restrictlist);
                if (fk_sel < 1){
                    fk = (FKSelecInfo*) palloc(sizeof(FKSelecInfo));
                    fk->next = info->base_rels[i].fk_selecs;
                    info->base_rels[i].fk_selecs = fk;
                    fk->other_baserel = j;
                    fk->sel = fk_sel;
                    info->n_fk_selecs++;
                }
            }
            j++;
        }
        i++;
    }
}

/*
 * gpuqo
 *	  solution of the query optimization problem
 *	  using GPU acceleration (CUDA)
 */

RelOptInfo *
gpuqo(PlannerInfo *root, int n_rels, List *initial_rels)
{
    ListCell* lc;
    int i;
    RelOptInfo* rel;
    GpuqoPlannerInfo* info;
    QueryTree* query_tree;

#ifdef OPTIMIZER_DEBUG
    printf("Hello, here is gpuqo!\n");
#endif

    info = (GpuqoPlannerInfo*) palloc(sizeof(GpuqoPlannerInfo));
    
    info->base_rels = (BaseRelation*) palloc(n_rels * sizeof(BaseRelation));

    info->n_rels = n_rels;
    info->eq_classes = NULL;
    info->n_fk_selecs = 0;
    info->n_eq_classes = 0;
    info->n_eq_class_sels = 0;

    i = 0;
    foreach(lc, initial_rels){
        rel = (RelOptInfo *) lfirst(lc);
        info->base_rels[i++] = makeBaseRelation(rel, root);
    }
    info->edge_table = makeEdgeTable(root, n_rels);
    fillSelectivityInformation(root, initial_rels, info, n_rels);

#ifdef GPUQO_INFO
    printEdges(info);
#endif

    query_tree = gpuqo_run(gpuqo_algorithm, info);
    
    pfree(info->base_rels);
    pfree(info->edge_table);
    pfree(info);
    // TODO free lists
    
#ifdef GPUQO_INFO
    printQueryTree(query_tree, 2);
    printf("gpuqo cost is %f\n", query_tree->cost);
#endif

	return queryTree2Plan(query_tree, n_rels, root, n_rels, initial_rels);
}
