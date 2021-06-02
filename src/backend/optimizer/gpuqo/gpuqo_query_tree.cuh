/*-------------------------------------------------------------------------
 *
 * gpuqo_query_tree.cuh
 *	  declaration of QueryTree<BitmapsetN>-related functions used in cu files
 * 
 * src/include/optimizer/gpuqo_query_tree.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_QUERY_TREE_CUH
#define GPUQO_QUERY_TREE_CUH

#include <thrust/device_vector.h>

#include "gpuqo.cuh"
#include "gpuqo_hashtable.cuh"

template <typename BitmapsetN, typename Container>
void dpsize_buildQueryTree(uint_t<BitmapsetN> idx, Container &gpu_memo_vals, QueryTree<BitmapsetN> **qt)
{
    JoinRelationDpsize<BitmapsetN> jr = gpu_memo_vals[idx];

    (*qt) = (QueryTree<BitmapsetN>*) malloc(sizeof(QueryTree<BitmapsetN>));
    (*qt)->id = jr.id;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr.rows;
    (*qt)->cost = jr.cost;
    (*qt)->width = jr.width;

    if (jr.left_rel_id.empty() && jr.left_rel_id.empty()){ // leaf
        return;
    }

    if (jr.left_rel_id.empty() || jr.right_rel_id.empty()){ // error
        printf("ERROR in buildQueryTree: %u has children %u and %u\n",
                jr.id.toUint(), 
                jr.left_rel_id.toUint(), jr.right_rel_id.toUint());
        return;
    }

    dpsize_buildQueryTree<BitmapsetN, Container>(jr.left_rel_idx, gpu_memo_vals, &((*qt)->left));
    dpsize_buildQueryTree<BitmapsetN, Container>(jr.right_rel_idx, gpu_memo_vals, &((*qt)->right));
}

template <typename BitmapsetN, typename Container>
void dpsub_buildQueryTree(BitmapsetN id, Container &gpu_memo_vals, QueryTree<BitmapsetN> **qt)
{
    JoinRelation<BitmapsetN> jr = gpu_memo_vals.get(id);

    (*qt) = (QueryTree<BitmapsetN>*) malloc(sizeof(QueryTree<BitmapsetN>));
    (*qt)->id = id;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr.rows;
    (*qt)->cost = jr.cost;
    (*qt)->width = jr.width;

    if (jr.left_rel_id.empty() && jr.left_rel_id.empty()){ // leaf
        return;
    }

    if (jr.left_rel_id == 0 || jr.right_rel_id == 0){ // error
        printf("ERROR in buildQueryTree: %u has children %u and %u\n",
                id.toUint(), jr.left_rel_id.toUint(), jr.right_rel_id.toUint());
        return;
    }

    dpsub_buildQueryTree<BitmapsetN, Container>(jr.left_rel_id, gpu_memo_vals, &((*qt)->left));
    dpsub_buildQueryTree<BitmapsetN, Container>(jr.right_rel_id, gpu_memo_vals, &((*qt)->right));
}

#endif							/* GPUQO_QUERY_TREE_CUH */
