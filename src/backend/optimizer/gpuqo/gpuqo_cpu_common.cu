/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpsize.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsize.cu
 *
 *-------------------------------------------------------------------------
 */

#include <list>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_common.cuh"
#include "gpuqo_dependency_buffer.cuh"

template<typename BitmapsetN> 
void build_query_tree(JoinRelationCPU<BitmapsetN> *jr, 
                        memo_t<BitmapsetN>  &memo, QueryTree<BitmapsetN> **qt)
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

    build_query_tree<BitmapsetN>(jr->left_rel_ptr, memo, &((*qt)->left));
    build_query_tree<BitmapsetN>(jr->right_rel_ptr, memo, &((*qt)->right));
}

template void build_query_tree<Bitmapset32>(JoinRelationCPU<Bitmapset32> *, memo_t<Bitmapset32>&, QueryTree<Bitmapset32>**);
template void build_query_tree<Bitmapset64>(JoinRelationCPU<Bitmapset64> *, memo_t<Bitmapset64>&, QueryTree<Bitmapset64>**);

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
    join_rel->cost = INFF;
    join_rel->rows = INFF;

#ifdef USE_ASSERT_CHECKING
    join_rel->referenced = false;
#endif

    return join_rel;
}

// explicitly instantiate template implementations
// by doing so I avoid defining the template in the header file
template JoinRelationCPU<Bitmapset32> *build_join_relation<JoinRelationCPU<Bitmapset32> >(JoinRelationCPU<Bitmapset32> &, JoinRelationCPU<Bitmapset32> &);
template JoinRelationCPU<Bitmapset64> *build_join_relation<JoinRelationCPU<Bitmapset64> >(JoinRelationCPU<Bitmapset64> &, JoinRelationCPU<Bitmapset64> &);
template JoinRelationDPE<Bitmapset32> *build_join_relation<JoinRelationDPE<Bitmapset32> >(JoinRelationDPE<Bitmapset32> &, JoinRelationDPE<Bitmapset32> &);
template JoinRelationDPE<Bitmapset64> *build_join_relation<JoinRelationDPE<Bitmapset64> >(JoinRelationDPE<Bitmapset64> &, JoinRelationDPE<Bitmapset64> &);

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

    return join_rel;
}

// explicitly instantiate template implementations
// by doing so I avoid defining the template in the header file
template JoinRelationCPU<Bitmapset32> *make_join_relation<Bitmapset32, JoinRelationCPU<Bitmapset32> >(JoinRelationCPU<Bitmapset32> &, JoinRelationCPU<Bitmapset32> &, GpuqoPlannerInfo<Bitmapset32>*);
template JoinRelationCPU<Bitmapset64> *make_join_relation<Bitmapset64, JoinRelationCPU<Bitmapset64> >(JoinRelationCPU<Bitmapset64> &, JoinRelationCPU<Bitmapset64> &, GpuqoPlannerInfo<Bitmapset64>*);
template JoinRelationDPE<Bitmapset32> *make_join_relation<Bitmapset32, JoinRelationDPE<Bitmapset32> >(JoinRelationDPE<Bitmapset32> &, JoinRelationDPE<Bitmapset32> &, GpuqoPlannerInfo<Bitmapset32>*);
template JoinRelationDPE<Bitmapset64> *make_join_relation<Bitmapset64, JoinRelationDPE<Bitmapset64> >(JoinRelationDPE<Bitmapset64> &, JoinRelationDPE<Bitmapset64> &, GpuqoPlannerInfo<Bitmapset64>*);

template<typename BitmapsetN, typename JR>
bool do_join(int level, JR* &join_rel, JR &left_rel, 
            JR &right_rel, 
            GpuqoPlannerInfo<BitmapsetN>* info, memo_t<BitmapsetN> &memo){
    join_rel = make_join_relation<BitmapsetN,JR>(left_rel, right_rel, info);

    auto find_iter = memo.find(join_rel->id);
    if (find_iter != memo.end()){
        JoinRelationCPU<BitmapsetN>* old_jr = find_iter->second;  

#ifdef USE_ASSERT_CHECKING
        Assert(!old_jr->referenced);
#endif
        if (join_rel->cost < old_jr->cost){
            *old_jr = *join_rel;
        }
        delete join_rel;
        return false;
    } else{
        memo.insert(std::make_pair(join_rel->id, join_rel));
        return true;
    }
}

// explicitly instantiate template implementations
// by doing so I avoid defining the template in the header file
template bool do_join<Bitmapset32,JoinRelationCPU<Bitmapset32> >(int, JoinRelationCPU<Bitmapset32> *&, JoinRelationCPU<Bitmapset32> &, JoinRelationCPU<Bitmapset32> &, GpuqoPlannerInfo<Bitmapset32>*, memo_t<Bitmapset32> &);
template bool do_join<Bitmapset64,JoinRelationCPU<Bitmapset64> >(int, JoinRelationCPU<Bitmapset64> *&, JoinRelationCPU<Bitmapset64> &, JoinRelationCPU<Bitmapset64> &, GpuqoPlannerInfo<Bitmapset64>*, memo_t<Bitmapset64> &);
template bool do_join<Bitmapset32,JoinRelationDPE<Bitmapset32> >(int, JoinRelationDPE<Bitmapset32> *&, JoinRelationDPE<Bitmapset32> &, JoinRelationDPE<Bitmapset32> &, GpuqoPlannerInfo<Bitmapset32>*, memo_t<Bitmapset32> &);
template bool do_join<Bitmapset64,JoinRelationDPE<Bitmapset64> >(int, JoinRelationDPE<Bitmapset64> *&, JoinRelationDPE<Bitmapset64> &, JoinRelationDPE<Bitmapset64> &, GpuqoPlannerInfo<Bitmapset64>*, memo_t<Bitmapset64> &);
