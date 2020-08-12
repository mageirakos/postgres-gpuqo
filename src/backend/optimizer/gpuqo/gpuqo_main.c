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

GpuqoAlgorithm gpuqo_algorithm;

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root);
FixedBitMask bitmapset2FixedBitMask(Bitmapset* set);
void printQueryTree(QueryTree* qt, int indent);
RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int number_of_rels, List *initial_rels);
void makeEdgeTable(PlannerInfo *root, int number_of_rels, List *initial_rels, BaseRelation* base_rels, EdgeInfo* edge_table);

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

RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int number_of_rels, List *initial_rels){
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
        left_rel = queryTree2Plan(qt->left, level-1, root, number_of_rels, initial_rels);
        right_rel = queryTree2Plan(qt->right, level-1, root, number_of_rels,initial_rels);
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
            if (level != number_of_rels)
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

FixedBitMask bitmapset2FixedBitMask(Bitmapset* set){
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
    
    baserel.id = bitmapset2FixedBitMask(rel->relids);

    baserel.edges = 0;
    foreach(lc, root->eq_classes){
        EquivalenceClass *ec = (EquivalenceClass *) lfirst(lc);
        if (bms_overlap(rel->relids, ec->ec_relids)){
            baserel.edges |= bitmapset2FixedBitMask(ec->ec_relids);
        }
    }

    // remove itself if present
    baserel.edges &= ~baserel.id;

    return baserel;
}

void makeEdgeTable(PlannerInfo *root, int number_of_rels, List *initial_rels, BaseRelation* base_rels, EdgeInfo* edge_table){
    ListCell* lc_inner;
    ListCell* lc_outer;
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
            
            if (base_rels[i].edges & bitmapset2FixedBitMask(rel_inner->relids)){
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
            } else {
                edge_info.sel = 1; // cross-join selectivity
            }
            edge_table[i*number_of_rels+j] = edge_info;
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
gpuqo(PlannerInfo *root, int number_of_rels, List *initial_rels)
{
    ListCell* lc;
    int i;
    RelOptInfo* rel;
    BaseRelation* baserels;
    EdgeInfo* edge_table;
    QueryTree* query_tree;
	
#ifdef OPTIMIZER_DEBUG
    printf("Hello, here is gpuqo!\n");
#endif

    baserels = (BaseRelation*) malloc(number_of_rels * sizeof(BaseRelation));
    edge_table = (EdgeInfo*) malloc(number_of_rels * number_of_rels * sizeof(EdgeInfo));

    i = 0;
    foreach(lc, initial_rels){
        rel = (RelOptInfo *) lfirst(lc);
        baserels[i++] = makeBaseRelation(rel, root);
    }
    makeEdgeTable(root, number_of_rels, initial_rels, baserels, edge_table);

    switch (gpuqo_algorithm)
    {
    case GPUQO_DPSIZE:
        query_tree = gpuqo_dpsize(baserels, number_of_rels, edge_table);
        break;
    case GPUQO_CPU_DPSIZE:
        query_tree = gpuqo_cpu_dpsize(baserels, number_of_rels, edge_table);
        break;
    }
    

    free(baserels);
    free(edge_table);

#ifdef OPTIMIZER_DEBUG
    printQueryTree(query_tree, 2);
#endif

	return queryTree2Plan(query_tree, number_of_rels, root, number_of_rels, initial_rels);
}
