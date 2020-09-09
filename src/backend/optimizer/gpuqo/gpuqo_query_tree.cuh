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

template<typename T>
void buildQueryTree(uint64_t idx, T &gpu_memo_vals, QueryTree **qt);

#endif							/* GPUQO_QUERY_TREE_CUH */
