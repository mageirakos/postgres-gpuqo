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

template<typename BitmapsetN>
struct JoinRelationCPU : public JoinRelationDetailed<BitmapsetN>{
	struct JoinRelationCPU<BitmapsetN>* left_rel_ptr;
	struct JoinRelationCPU<BitmapsetN>* right_rel_ptr;
#ifdef USE_ASSERT_CHECKING
	bool referenced;
#endif
};

template<typename BitmapsetN>
using vector_list_t = std::vector< std::list<JoinRelationCPU<BitmapsetN>*> >;

template<typename BitmapsetN>
using memo_t = std::unordered_map<BitmapsetN, JoinRelationCPU<BitmapsetN>*>;

// forward declaration
template<typename BitmapsetN>
class CPUAlgorithm;

template<typename BitmapsetN>
class CPUJoinFunction {
protected:
	GpuqoPlannerInfo<BitmapsetN>* info;
	memo_t<BitmapsetN>* memo;
	CPUAlgorithm<BitmapsetN>* alg;
public:
	CPUJoinFunction() {}

	CPUJoinFunction(GpuqoPlannerInfo<BitmapsetN>* _info, 
					memo_t<BitmapsetN>* _memo, CPUAlgorithm<BitmapsetN>* _alg) 
		: info(_info), memo(_memo), alg(_alg) {}
	
	virtual void operator()(int level, bool try_swap, 
		JoinRelationCPU<BitmapsetN> &left_rel,
		JoinRelationCPU<BitmapsetN> &right_rel) {}
};

template<typename BitmapsetN>
class CPUAlgorithm{
protected:
	GpuqoPlannerInfo<BitmapsetN>* info;
	memo_t<BitmapsetN>* memo;
	CPUJoinFunction<BitmapsetN> *join;
public:
	virtual void init(GpuqoPlannerInfo<BitmapsetN>* _info, 
					memo_t<BitmapsetN>* _memo,
					CPUJoinFunction<BitmapsetN> *_join)
	{
		info = _info;
		memo = _memo;
		join = _join;
	}

	virtual bool check_join(int level, JoinRelationCPU<BitmapsetN> &left_rel,
			JoinRelationCPU<BitmapsetN> &right_rel) {return true;}

	virtual void post_join(int level,  bool new_rel, 		
		   JoinRelationCPU<BitmapsetN> &join_rel,
		   JoinRelationCPU<BitmapsetN> &left_rel, 
		   JoinRelationCPU<BitmapsetN> &right_rel) {}

	virtual void enumerate() {}
};

template<typename BitmapsetN>
extern void build_query_tree(JoinRelationCPU<BitmapsetN> *jr,
						memo_t<BitmapsetN> &memo, QueryTree<BitmapsetN> **qt);

template<typename JR>
JR* build_join_relation(JR &left_rel, JR &right_rel);

template<typename BitmapsetN, typename JR>
JR* make_join_relation(JR &left_rel, JR &right_rel, GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN, typename JR>
bool do_join(int level, JR* &join_rel, JR &left_rel, JR &right_rel, 
            GpuqoPlannerInfo<BitmapsetN>* info, memo_t<BitmapsetN> &memo);

#endif							/* GPUQO_CPU_COMMON_CUH */
