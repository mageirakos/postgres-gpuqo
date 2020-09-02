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

#include "postgres.h"
#include "miscadmin.h"

#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/gpuqo.h"

int gpuqo_algorithm;

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root);
Bitmapset64 convertBitmapset(Bitmapset* set);
void printQueryTree(QueryTree* qt, int indent);
void printEdges(GpuqoPlannerInfo* info);
RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int n_rels, List *initial_rels);
void fillEdgeTable(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfo* info);

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
    printf("%llu (rows=%.0f, cost=%.2f)\n", qt->id, qt->rows, qt->cost);

    printQueryTree(qt->left, indent + 2);
    printQueryTree(qt->right, indent + 2);
}

void printEdges(GpuqoPlannerInfo* info){
    for (int i = 0; i < info->n_rels; i++){
        RelationID edges = info->base_rels[i].edges;
        printf("%d:", i+1);
        while (edges != BMS64_EMPTY){
            int idx = BMS64_LOWEST_POS(edges)-1;
            EdgeInfo* edge = &info->edge_table[i*info->n_rels+idx-1];
            printf(" %d(%.4f,%s)", idx, edge->sel, edge->has_index ? "I": "");
            edges = BMS64_UNSET(edges, idx);
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
        printf("WARNING: Found NULL RelOptInfo*\n");
    }

    // clean-up the query tree
    free(qt);

    return this_rel;
}

Bitmapset64 convertBitmapset(Bitmapset* set){
    if (set->nwords > 1){
        printf("WARNING: only relids of 64 bits are supported!\n");
    }
    return set->words[0];
}

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root){
    ListCell* lc;
    BaseRelation baserel;
    

    baserel.rows = rel->rows;
    baserel.tuples = rel->tuples;
    
    baserel.id = convertBitmapset(rel->relids);

    baserel.edges = 0;
    foreach(lc, root->eq_classes){
        EquivalenceClass *ec = (EquivalenceClass *) lfirst(lc);

        if (list_length(ec->ec_members) <= 1)
			continue;

        if (bms_overlap(rel->relids, ec->ec_relids)){
            baserel.edges = BMS64_UNION(baserel.edges, convertBitmapset(ec->ec_relids));
        }
    }

    // remove itself if present
    baserel.edges = BMS64_DIFFERENCE(baserel.edges, baserel.id);

    return baserel;
}

void fillEdgeTable(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfo* info){
    ListCell* lc_inner;
    ListCell* lc_outer;
    ListCell* lc_inner_path;
    EdgeInfo edge_info;
    RelOptInfo* joinrel;
    List	   *restrictlist = NULL;
    int i, j;

    i = 0;
    foreach(lc_outer, initial_rels){
        RelOptInfo* rel_outer = (RelOptInfo*) lfirst(lc_outer);
        j = 0;
        foreach(lc_inner, initial_rels){
            RelOptInfo* rel_inner = (RelOptInfo*) lfirst(lc_inner);
            
            if (BMS64_INTERSECTS(info->base_rels[i].edges, convertBitmapset(rel_inner->relids))){
                double sel;
                SpecialJoinInfo sjinfo;
                
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

                joinrel = build_join_rel(root,
                                        bms_union(rel_outer->relids, 
                                                  rel_inner->relids),
                                        rel_outer,
                                        rel_inner,
                                        &sjinfo,
                                        &restrictlist);

                // this is just a quick'n'dirty way, in reality the number of
                // rows is clamped so I might underestimate the selectivity
                sel = joinrel->rows / (rel_inner->rows*rel_outer->rows);
                edge_info.sel = sel < 1 ? sel : 1;

                edge_info.has_index = false;
                foreach(lc_inner_path, rel_inner->pathlist){
                    Path* path = (Path*) lfirst(lc_inner_path);
                    if (path->pathtype == T_IndexScan){
                        if (bms_num_members(PATH_REQ_OUTER(path)) == 1 
                                && bms_overlap(PATH_REQ_OUTER(path),
                                    rel_outer->relids
                                )){
                            edge_info.has_index = true;
                            break;
                        }
                    }
                }
            } else {
                edge_info.sel = 1; // cross-join selectivity
                edge_info.has_index = false;
            }
            info->edge_table[i*info->n_rels+j] = edge_info;
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

    info = (GpuqoPlannerInfo*) gpuqo_malloc(sizeof(GpuqoPlannerInfo));
    info->base_rels = (BaseRelation*) gpuqo_malloc(n_rels * sizeof(BaseRelation));
    info->n_rels = n_rels;
    info->edge_table = (EdgeInfo*) gpuqo_malloc(n_rels * n_rels * sizeof(EdgeInfo));

    i = 0;
    foreach(lc, initial_rels){
        rel = (RelOptInfo *) lfirst(lc);
        info->base_rels[i++] = makeBaseRelation(rel, root);
    }
    fillEdgeTable(root, initial_rels, info);
    
    // prefetch to GPU (noop if on CPU)
    gpuqo_prefetch(info);

#ifdef OPTIMIZER_DEBUG
    printEdges(info);
#endif

    switch (gpuqo_algorithm)
    {
    case GPUQO_DPSIZE:
        query_tree = gpuqo_dpsize(info);
        break;
    case GPUQO_DPSUB:
        query_tree = gpuqo_dpsub(info);
        break;
    case GPUQO_CPU_DPSIZE:
        query_tree = gpuqo_cpu_dpsize(info);
        break;
    case GPUQO_CPU_DPSUB:
        query_tree = gpuqo_cpu_dpsub(info);
        break;
    case GPUQO_CPU_DPCCP:
        query_tree = gpuqo_cpu_dpccp(info);
        break;
    case GPUQO_DPE_DPSIZE:
        query_tree = gpuqo_dpe_dpsize(info);
        break;
    case GPUQO_DPE_DPSUB:
        query_tree = gpuqo_dpe_dpsub(info);
        break;
    case GPUQO_DPE_DPCCP:
        query_tree = gpuqo_dpe_dpccp(info);
        break;
    default: 
        // impossible branch but without it the compiler complains
        query_tree = NULL;
        break;
    }
    
    gpuqo_free(info->base_rels);
    gpuqo_free(info->edge_table);
    gpuqo_free(info);
    
#ifdef OPTIMIZER_DEBUG
    printQueryTree(query_tree, 2);
#endif

	return queryTree2Plan(query_tree, n_rels, root, n_rels, initial_rels);
}
