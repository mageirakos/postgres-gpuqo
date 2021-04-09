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

#ifdef OPTIMIZER_DEBUG
#ifndef GPUQO_INFO
#define GPUQO_INFO
#endif
#endif

int gpuqo_algorithm;

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root);
EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels);
Bitmapset32 convertBitmapset(Bitmapset* set);
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
    printf("%u (rows=%.0f, cost=%.2f)\n", qt->id, qt->rows, qt->cost);

    printQueryTree(qt->left, indent + 2);
    printQueryTree(qt->right, indent + 2);
}

void printEdges(GpuqoPlannerInfo* info){
    printf("\nEdges:\n");
    for (int i = 0; i < info->n_rels; i++){
        RelationID edges = info->edge_table[i];
        printf("%d:", i+1);
        while (edges != BMS32_EMPTY){
            int idx = BMS32_LOWEST_POS(edges)-1;
            EqClassInfo *ec = info->eq_classes;
            
            printf(" %d ", idx);

            if (ec != NULL){
                printf("(");
                while (ec != NULL){
                    if (BMS32_IS_SET(ec->relids, i+1) && BMS32_IS_SET(ec->relids, idx)){
                        int idx;
                        int size = BMS32_SIZE(ec->relids);
                        int idx_l = BMS32_SIZE(
                            BMS32_INTERSECTION(
                                BMS32_SET_ALL_LOWER(BMS32_NTH(i+1)),
                                ec->relids
                            )
                        );
                        int idx_r = BMS32_SIZE(
                            BMS32_INTERSECTION(
                                BMS32_SET_ALL_LOWER(BMS32_NTH(idx)),
                                ec->relids
                            )
                        );
                        if (idx_l > idx_r){
                            int tmp = idx_l;
                            idx_l = idx_r;
                            idx_r = tmp;
                        }

                        idx = idx_l*size - idx_l*(idx_l+1)/2 + (idx_r-idx_l-1);
                        printf("%u: %f, ", ec->relids, ec->sels[idx]);
                    }
                    ec = ec->next;
                }
                printf(");");
            }

            edges = BMS32_UNSET(edges, idx);
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
            if (base_rel->relids->words[0] & qt->id){
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
        printf("WARNING: Found NULL RelOptInfo*: %u (%u, %u)\n", 
                qt->id, 
                qt->left ? qt->left->id : 0, 
                qt->right ? qt->right->id : 0
        );
    }

    // clean-up the query tree
    free(qt);

    return this_rel;
}

Bitmapset32 convertBitmapset(Bitmapset* set){
    if (set->nwords > 1){
        printf("WARNING: only relids of 32 bits are supported!\n");
    }
    if (set->words[0] & 0xFFFFFFFF00000000ULL){
        printf("WARNING: only relids of 32 bits are supported!\n");
    }
    return (Bitmapset32)(set->words[0] & 0xFFFFFFFFULL);
}

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root){
    BaseRelation baserel;
    
    baserel.rows = rel->rows;
    baserel.tuples = rel->tuples;
    baserel.id = convertBitmapset(rel->relids);
    baserel.fk_selecs = NULL;

    return baserel;
}

EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels){
    ListCell* lc;

    EdgeMask* edge_table = malloc(sizeof(EdgeMask)*n_rels);
    memset(edge_table, 0, sizeof(EdgeMask)*n_rels);
    foreach(lc, root->eq_classes){
        EquivalenceClass *ec = (EquivalenceClass *) lfirst(lc);
        EdgeMask ec_relids = convertBitmapset(ec->ec_relids);

        if (list_length(ec->ec_members) <= 1)
			continue;

        for (int i=1; i <= n_rels; i++){
            RelationID baserelid = BMS32_NTH(i);
            if (BMS32_INTERSECTS(ec_relids, baserelid)){
                edge_table[i-1] = BMS32_UNION(
                        edge_table[i-1],
                        BMS32_DIFFERENCE(ec_relids, baserelid)
                );
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

    info->indexed_edge_table = (EdgeMask*) malloc(n_rels * sizeof(EdgeMask));
    memset(info->indexed_edge_table, 0, n_rels * sizeof(EdgeMask));

    i = 0;
    foreach(lc_outer, initial_rels){
        RelOptInfo* rel_outer = (RelOptInfo*) lfirst(lc_outer);
        j = 0;
        foreach(lc_inner, initial_rels){
            RelOptInfo* rel_inner = (RelOptInfo*) lfirst(lc_inner);
            
            if (BMS32_INTERSECTS(info->edge_table[i], convertBitmapset(rel_inner->relids))){
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
                        if (ec->relids == convertBitmapset(rinfo->parent_ec->ec_relids))
                            break;
                        ec = ec->next;

                    };

                    if (ec == NULL){
                        ec = (EqClassInfo*) malloc(sizeof(EqClassInfo));
                        ec->next = info->eq_classes;
                        info->eq_classes = ec;
                        ec->relids = convertBitmapset(rinfo->parent_ec->ec_relids);
                        size = BMS32_SIZE(ec->relids);
                        ec->sels = (float*) malloc(sizeof(float)*size*size);
                        info->n_eq_classes++;
                        info->n_eq_class_sels += size*(size-1)/2;
                    } else {
                        size = BMS32_SIZE(ec->relids);
                    }

                    idx_l = BMS32_SIZE(
                        BMS32_INTERSECTION(
                            BMS32_SET_ALL_LOWER(BMS32_NTH(i+1)),
                            ec->relids
                        )
                    );
                    idx_r = BMS32_SIZE(
                        BMS32_INTERSECTION(
                            BMS32_SET_ALL_LOWER(BMS32_NTH(j+1)),
                            ec->relids
                        )
                    );
                    if (idx_l > idx_r){
                        int tmp = idx_l;
                        idx_l = idx_r;
                        idx_r = tmp;
                    }
                    idx = idx_l*size - idx_l*(idx_l+1)/2 + (idx_r-idx_l-1);
                    ec->sels[idx] = rinfo->norm_selec;
                }

                foreach(lc_inner_path, rel_inner->pathlist){
                    Path* path = (Path*) lfirst(lc_inner_path);
                    if (path->pathtype == T_IndexScan){
                        if (bms_num_members(PATH_REQ_OUTER(path)) == 1 
                                && bms_overlap(PATH_REQ_OUTER(path),
                                    rel_outer->relids
                                )){
                            info->indexed_edge_table[j] = BMS32_SET(info->indexed_edge_table[j], i+1);
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
                    fk = (FKSelecInfo*) malloc(sizeof(FKSelecInfo));
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

    info = (GpuqoPlannerInfo*) malloc(sizeof(GpuqoPlannerInfo));
    
    info->base_rels = (BaseRelation*) malloc(n_rels * sizeof(BaseRelation));

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
    
    free(info->base_rels);
    free(info->edge_table);
    free(info);
    // TODO free lists
    
#ifdef GPUQO_INFO
    printQueryTree(query_tree, 2);
    printf("gpuqo cost is %f\n", query_tree->cost);
#endif

	return queryTree2Plan(query_tree, n_rels, root, n_rels, initial_rels);
}
