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

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"

typedef std::vector< std::list<JoinRelation*> > vector_list_t;
typedef std::unordered_map<RelationID, JoinRelation*> memo_t;

typedef struct extra_t{
	// algorithm extras 
	void* alg;

	// implementation extras (e.g. depbuf in DPE)
	void* impl;
} extra_t;

struct DPCPUAlgorithm;

typedef void (*join_f)(int level, bool try_swap, JoinRelation &left_rel,
					JoinRelation &right_rel, GpuqoPlannerInfo* info, 
					memo_t &memo, extra_t extra, 
					struct DPCPUAlgorithm algorithm);
typedef bool (*check_join_f)(int level, JoinRelation &left_rel,
 					JoinRelation &right_rel, GpuqoPlannerInfo* info, 
					memo_t &memo, extra_t extra);
typedef void (*post_join_f)(int level,  bool new_rel, JoinRelation &join_rel,
					JoinRelation &left_rel, JoinRelation &right_rel,
					GpuqoPlannerInfo* info, memo_t &memo, extra_t extra);
typedef void (*enumerate_f)(GpuqoPlannerInfo* info, join_f join_function, memo_t &memo, extra_t extra, struct DPCPUAlgorithm algorithm);
typedef void (*init_f)(GpuqoPlannerInfo* info, memo_t &memo, extra_t &extra);
typedef void (*teardown_f)(GpuqoPlannerInfo* info, memo_t &memo, extra_t extra);

typedef struct DPCPUAlgorithm{
	init_f init_function;
	enumerate_f enumerate_function;
	check_join_f check_join_function;
	post_join_f post_join_function;
	teardown_f teardown_function;
} DPCPUAlgorithm;

extern void build_query_tree(JoinRelation *jr, memo_t &memo, QueryTree **qt);

template<typename T>
T* build_join_relation(T &left_rel,T &right_rel);

template<typename T>
T* make_join_relation(T &left_rel,T &right_rel, GpuqoPlannerInfo* info);

template<typename T>
bool do_join(int level, T* &join_rel, T &left_rel, T &right_rel, 
            GpuqoPlannerInfo* info, memo_t &memo, extra_t extra);

#endif							/* GPUQO_CPU_COMMON_CUH */
