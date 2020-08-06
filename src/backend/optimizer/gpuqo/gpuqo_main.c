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

#include "optimizer/paths.h"
#include "optimizer/gpuqo.h"

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root);
FixedBitMask bitmapset2FixedBitMask(Bitmapset* set);
void printQueryTree(QueryTree* qt, int indent);
RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int number_of_rels, List *initial_rels);

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
    printf("%d (rows=%d, cost=%.2f)\n", qt->id, qt->rows, qt->cost);

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
    QueryTree* query_tree;
	
#ifdef OPTIMIZER_DEBUG
    printf("Hello, here is gpuqo!\n");
#endif

    baserels = (BaseRelation*) malloc(number_of_rels * sizeof(BaseRelation));

    i = 0;
    foreach(lc, initial_rels){
        rel = (RelOptInfo *) lfirst(lc);
        baserels[i++] = makeBaseRelation(rel, root);
    }

    query_tree = gpuqo_dpsize(baserels, number_of_rels);

    free(baserels);

#ifdef OPTIMIZER_DEBUG
    printQueryTree(query_tree, 2);
#endif

	return queryTree2Plan(query_tree, number_of_rels, root, number_of_rels, initial_rels);
}
