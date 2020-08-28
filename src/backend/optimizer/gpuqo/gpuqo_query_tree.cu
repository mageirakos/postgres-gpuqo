/*-------------------------------------------------------------------------
 *
 * gpuqo_query_tree.cu
 *	  definition of QueryTree-related functions used in cu files
 *
 * src/include/optimizer/gpuqo_query_tree.cu
 *
 *-------------------------------------------------------------------------
 */

#include <optimizer/gpuqo_query_tree.cuh>

template<typename T>
void buildQueryTree(uint64_t idx, T &gpu_memo_vals, QueryTree **qt)
{
    JoinRelation jr = gpu_memo_vals[idx];

    (*qt) = (QueryTree*) malloc(sizeof(QueryTree));
    (*qt)->id = jr.id;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr.rows;
    (*qt)->cost = jr.cost;

    if (jr.left_relation_id == 0 && jr.right_relation_id == 0)
    return;

    buildQueryTree<T>(jr.left_relation_idx, gpu_memo_vals, &((*qt)->left));
    buildQueryTree<T>(jr.right_relation_idx, gpu_memo_vals, &((*qt)->right));
}

template void buildQueryTree< thrust::device_vector<JoinRelation> >(uint64_t idx, thrust::device_vector<JoinRelation> &gpu_memo_vals, QueryTree **qt);
template void buildQueryTree<uninit_device_vector_joinrel>(uint64_t idx, uninit_device_vector_joinrel &gpu_memo_vals, QueryTree **qt);
