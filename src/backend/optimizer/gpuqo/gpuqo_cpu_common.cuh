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
using hashtable_memo_t = std::unordered_map<BitmapsetN, JoinRelationCPU<BitmapsetN>*>;

// forward declaration
template<typename BitmapsetN, typename memo_t>
class CPUAlgorithm;

template<typename BitmapsetN, typename memo_t>
class CPUJoinFunction {
protected:
	GpuqoPlannerInfo<BitmapsetN>* info;
	memo_t* memo;
	CPUAlgorithm<BitmapsetN, memo_t>* alg;
public:
	CPUJoinFunction() {}

	CPUJoinFunction(GpuqoPlannerInfo<BitmapsetN>* _info, 
					memo_t* _memo, CPUAlgorithm<BitmapsetN, memo_t>* _alg) 
		: info(_info), memo(_memo), alg(_alg) {}
	
	virtual JoinRelationCPU<BitmapsetN> *operator()(int level, bool try_swap, 
		JoinRelationCPU<BitmapsetN> &left_rel,
		JoinRelationCPU<BitmapsetN> &right_rel) {return NULL;}
};

template<typename BitmapsetN, typename memo_t>
class CPUAlgorithm{
protected:
	GpuqoPlannerInfo<BitmapsetN>* info;
	memo_t* memo;
	CPUJoinFunction<BitmapsetN, memo_t> *join;
#ifdef GPUQO_PRINT_N_JOINS
	std::atomic<uint_t<BitmapsetN> > n_joins;
	std::atomic<uint_t<BitmapsetN> > n_checks;
#endif
public:
	virtual void init(GpuqoPlannerInfo<BitmapsetN>* _info, 
					memo_t* _memo,
					CPUJoinFunction<BitmapsetN, memo_t> *_join)
	{
		info = _info;
		memo = _memo;
		join = _join;

#ifdef GPUQO_PRINT_N_JOINS
		n_joins = 0;
		n_checks = 0;
#endif
	}

	virtual bool check_join(int level, JoinRelationCPU<BitmapsetN> &left_rel,
			JoinRelationCPU<BitmapsetN> &right_rel) {return true;}

	virtual void post_join(int level,  bool new_rel, 		
		   JoinRelationCPU<BitmapsetN> &join_rel,
		   JoinRelationCPU<BitmapsetN> &left_rel, 
		   JoinRelationCPU<BitmapsetN> &right_rel) 
	{
#ifdef GPUQO_PRINT_N_JOINS
		n_joins++;
#endif
	}

	virtual void enumerate() {}

	uint_t<BitmapsetN> get_n_joins(){
		return n_joins.load();
	}

	uint_t<BitmapsetN> get_n_checks(){
		return n_checks.load();
	}
};

template<typename BitmapsetN, typename memo_t> 
void build_query_tree(JoinRelationCPU<BitmapsetN> *jr, 
                        memo_t &memo, QueryTree<BitmapsetN> **qt)
{
    if (jr == NULL){
        (*qt) = NULL;
        return;
    }
    
    (*qt) = new QueryTree<BitmapsetN>;
    (*qt)->id = jr->id;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr->rows;
    (*qt)->cost = jr->cost;
    (*qt)->width = jr->width;

    build_query_tree<BitmapsetN>(jr->left_rel_ptr, memo, &((*qt)->left));
    build_query_tree<BitmapsetN>(jr->right_rel_ptr, memo, &((*qt)->right));
}

/* build_join_relation
 *
 *	 Builds a join relation from left and right relations. Do not compute cost.
 */
template<typename JR>
JR* build_join_relation(JR &left_rel, JR &right_rel){
	// give possibility to user to interrupt
	// This function is called by all CPU functions so putting it here catches
	// all of them
	CHECK_FOR_INTERRUPTS();

	JR* join_rel = new JR;

	join_rel->id = left_rel.id | right_rel.id;
	join_rel->left_rel_id = left_rel.id;
	join_rel->left_rel_ptr = &left_rel;
	join_rel->right_rel_id = right_rel.id;
	join_rel->right_rel_ptr = &right_rel;
	join_rel->edges = left_rel.edges | right_rel.edges;
	join_rel->cost.total = INFF;
	join_rel->cost.startup = INFF;
	join_rel->rows = INFF;

#ifdef USE_ASSERT_CHECKING
	join_rel->referenced = false;
#endif

	return join_rel;
}
 
/* make_join_relation
 *
 *	 Builds and fills a join relation from left and right relations.
 */
template<typename BitmapsetN, typename JR>
JR* make_join_relation(JR &left_rel, 
						JR &right_rel, 
						GpuqoPlannerInfo<BitmapsetN>* info)
{

#ifdef USE_ASSERT_CHECKING
	left_rel.referenced = true;
	right_rel.referenced = true;
#endif

	JR* join_rel = build_join_relation<JR>(left_rel, right_rel);
	join_rel->rows = estimate_join_rows(left_rel.id, left_rel, right_rel.id, right_rel, info);
	join_rel->cost = calc_join_cost(left_rel.id, left_rel, right_rel.id, right_rel, join_rel->rows, info);
	join_rel->width = get_join_width(left_rel.id, left_rel, right_rel.id, right_rel, info);

	return join_rel;
}
 

template<typename BitmapsetN, typename JR, typename memo_t>
bool do_join(int level, JR* &join_rel, JR &left_rel, 
            JR &right_rel, 
            GpuqoPlannerInfo<BitmapsetN>* info, memo_t &memo){
    join_rel = make_join_relation<BitmapsetN,JR>(left_rel, right_rel, info);

    auto find_iter = memo.find(join_rel->id);
    if (find_iter != memo.end()){
        JoinRelationCPU<BitmapsetN>* old_jr = find_iter->second;  

#ifdef USE_ASSERT_CHECKING
        Assert(!old_jr->referenced);
#endif
        if (join_rel->cost.total < old_jr->cost.total){
            *old_jr = *join_rel;
        }
        delete join_rel;
        return false;
    } else{
        memo.insert(std::make_pair(join_rel->id, join_rel));
        return true;
    }
}

#endif							/* GPUQO_CPU_COMMON_CUH */
