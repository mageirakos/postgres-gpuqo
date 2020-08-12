/*-------------------------------------------------------------------------
 *
 * gpuqo.cuh
 *	  function prototypes and struct definitions for CUDA/Thrust code
 *
 * src/include/optimizer/gpuqo.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_COMMON_CUH
#define GPUQO_COMMON_CUH

#include <list>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"

typedef std::vector< std::list< std::pair<RelationID, JoinRelation*> > > vector_list_t;
typedef std::unordered_map<RelationID, JoinRelation*> memo_t;

struct DPCPUAlgorithm;

typedef void (*join_f)(int level, 
					RelationID left_id, JoinRelation &left_rel,
					RelationID right_id, JoinRelation &right_rel,
					BaseRelation* base_rels, EdgeInfo* edge_table,
					int number_of_rels, memo_t &memo, void* extra,
					struct DPCPUAlgorithm algorithm);
typedef bool (*check_join_f)(int level, 
					RelationID left_id, JoinRelation &left_rel,
					RelationID right_id, JoinRelation &right_rel,
					BaseRelation* base_rels, EdgeInfo* edge_table,
					int number_of_rels, memo_t &memo, void* extra);
typedef void (*post_join_f)(int level,  bool new_rel,
					RelationID join_id, JoinRelation &join_rel,
					RelationID left_id, JoinRelation &left_rel,
					RelationID right_id, JoinRelation &right_rel,
					BaseRelation* base_rels, EdgeInfo* edge_table,
					int number_of_rels, memo_t &memo, void* extra);
typedef void (*enumerate_f)(BaseRelation baserels[], int N, EdgeInfo edge_table[], join_f join_function, memo_t &memo, void* extra, struct DPCPUAlgorithm algorithm);
typedef void (*init_f)(BaseRelation baserels[], int N, EdgeInfo edge_table[], memo_t &memo, void** extra);
typedef void (*teardown_f)(BaseRelation baserels[], int N, EdgeInfo edge_table[], memo_t &memo, void* extra);

typedef struct DPCPUAlgorithm{
	init_f init_function;
	enumerate_f enumerate_function;
	check_join_f check_join_function;
	post_join_f post_join_function;
	teardown_f teardown_function;
} DPCPUAlgorithm;

extern QueryTree* gpuqo_cpu_generic(BaseRelation baserels[], int N, EdgeInfo edge_table[], DPCPUAlgorithm algorithm);

#endif							/* GPUQO_COMMON_CUH */
