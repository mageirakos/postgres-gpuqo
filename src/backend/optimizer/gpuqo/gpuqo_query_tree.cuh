/*-------------------------------------------------------------------------
 *
 * gpuqo_query_tree.cuh
 *	  declaration of QueryTree-related functions used in cu files
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


template <typename T, typename Container>
static
T buildQueryTree_get(Container v, uint32_t idx){
    return v[idx];
}

template<> 
JoinRelation buildQueryTree_get(HashTable32bit ht, uint32_t idx){
    return ht.get(idx);
}


template <typename Container>
void dpsize_buildQueryTree(uint32_t idx, Container &gpu_memo_vals, QueryTree **qt)
{
    JoinRelationDpsize jr = buildQueryTree_get<JoinRelationDpsize, Container>(gpu_memo_vals, idx);

    (*qt) = (QueryTree*) malloc(sizeof(QueryTree));
    (*qt)->id = jr.id;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr.rows;
    (*qt)->cost = jr.cost;

    if (jr.left_rel_id == 0 && jr.left_rel_id == 0){ // leaf
        return;
    }

    if (jr.left_rel_id == 0 || jr.right_rel_id == 0){ // error
        printf("ERROR in buildQueryTree: %u has children %u and %u\n",
                jr.id, jr.left_rel_id, jr.right_rel_id);
        return;
    }

    dpsize_buildQueryTree<Container>(jr.left_rel_idx, gpu_memo_vals, &((*qt)->left));
    dpsize_buildQueryTree<Container>(jr.left_rel_idx, gpu_memo_vals, &((*qt)->right));
}

template <typename Container>
void dpsub_buildQueryTree(RelationID id, Container &gpu_memo_vals, QueryTree **qt)
{
    JoinRelation jr = buildQueryTree_get<JoinRelation, Container>(gpu_memo_vals, id);

    (*qt) = (QueryTree*) malloc(sizeof(QueryTree));
    (*qt)->id = id;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr.rows;
    (*qt)->cost = jr.cost;

    if (jr.left_rel_id == 0 && jr.left_rel_id == 0){ // leaf
        return;
    }

    if (jr.left_rel_id == 0 || jr.right_rel_id == 0){ // error
        printf("ERROR in buildQueryTree: %u has children %u and %u\n",
                id, jr.left_rel_id, jr.right_rel_id);
        return;
    }

    dpsub_buildQueryTree<Container>(jr.left_rel_id, gpu_memo_vals, &((*qt)->left));
    dpsub_buildQueryTree<Container>(jr.right_rel_id, gpu_memo_vals, &((*qt)->right));
}

#endif							/* GPUQO_QUERY_TREE_CUH */
