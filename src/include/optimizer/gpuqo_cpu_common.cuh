/*-------------------------------------------------------------------------
 *
 * gpuqo_cpu_common.cuh
 *	  definition of common types and functions for all CPU implementations.
 *
 * src/include/optimizer/gpuqo_cpu_common.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_CPU_COMMON_CUH
#define GPUQO_CPU_COMMON_CUH

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

typedef std::vector< std::list<JoinRelation*> > vector_list_t;
typedef std::unordered_map<RelationID, JoinRelation*> memo_t;

struct DPCPUAlgorithm;

typedef void (*join_f)(int level, bool try_swap, JoinRelation &left_rel,
					JoinRelation &right_rel, BaseRelation* base_rels, 
					int n_rels, EdgeInfo* edge_table, memo_t &memo, 
					void* extra, struct DPCPUAlgorithm algorithm);
typedef bool (*check_join_f)(int level, JoinRelation &left_rel,
 					JoinRelation &right_rel, BaseRelation* base_rels, 
					int n_rels, EdgeInfo* edge_table, memo_t &memo, 
					void* extra);
typedef void (*post_join_f)(int level,  bool new_rel, JoinRelation &join_rel,
					JoinRelation &left_rel, JoinRelation &right_rel,
					BaseRelation* base_rels, int n_rels, 
					EdgeInfo* edge_table, memo_t &memo, void* extra);
typedef void (*enumerate_f)(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], join_f join_function, memo_t &memo, void* extra, struct DPCPUAlgorithm algorithm);
typedef void (*init_f)(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], memo_t &memo, void** extra);
typedef void (*teardown_f)(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], memo_t &memo, void* extra);

typedef struct DPCPUAlgorithm{
	init_f init_function;
	enumerate_f enumerate_function;
	check_join_f check_join_function;
	post_join_f post_join_function;
	teardown_f teardown_function;
} DPCPUAlgorithm;

extern void build_query_tree(JoinRelation *jr, memo_t &memo, QueryTree **qt);

template<typename T>
T* make_join_relation(T &left_rel,T &right_rel,
                                 BaseRelation* base_rels, int n_rels,
                                 EdgeInfo* edge_table);

template<typename T>
bool do_join(int level, T* &join_rel, T &left_rel, T &right_rel, 
            BaseRelation* base_rels, int n_rels, 
            EdgeInfo* edge_table, memo_t &memo, void* extra);

#endif							/* GPUQO_CPU_COMMON_CUH */
