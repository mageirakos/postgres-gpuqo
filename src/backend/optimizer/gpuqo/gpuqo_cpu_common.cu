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

void build_query_tree(JoinRelation *jr, memo_t &memo, QueryTree **qt)
{
    if (jr == NULL){
        (*qt) = NULL;
        return;
    }
    
    (*qt) = new QueryTree;
    (*qt)->id = jr->id;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr->rows;
    (*qt)->cost = jr->cost;

    build_query_tree(jr->left_relation_ptr, memo, &((*qt)->left));
    build_query_tree(jr->right_relation_ptr, memo, &((*qt)->right));
}

/* build_join_relation
 *
 *	 Builds a join relation from left and right relations. Do not compute cost.
 */
template<typename T>
T* build_join_relation(T &left_rel,T &right_rel){
    // give possibility to user to interrupt
    // This function is called by all CPU functions so putting it here catches
    // all of them
    CHECK_FOR_INTERRUPTS();

    T* join_rel = new T;

    join_rel->id = BMS32_UNION(left_rel.id, right_rel.id);
    join_rel->left_relation_id = left_rel.id;
    join_rel->left_relation_ptr = &left_rel;
    join_rel->right_relation_id = right_rel.id;
    join_rel->right_relation_ptr = &right_rel;
    join_rel->edges = BMS32_UNION(left_rel.edges, right_rel.edges);
    join_rel->cost = INFF;
    join_rel->rows = INFF;

#ifdef USE_ASSERT_CHECKING
    join_rel->referenced = false;
#endif

    return join_rel;
}

// explicitly instantiate template implementations
// by doing so I avoid defining the template in the header file
template JoinRelation *build_join_relation<JoinRelation>(JoinRelation &, JoinRelation &);
template JoinRelationDPE *build_join_relation<JoinRelationDPE>(JoinRelationDPE &, JoinRelationDPE &);

/* make_join_relation
 *
 *	 Builds and fills a join relation from left and right relations.
 */
template<typename T>
T* make_join_relation(T &left_rel,T &right_rel, GpuqoPlannerInfo* info){

#ifdef USE_ASSERT_CHECKING
    left_rel.referenced = true;
    right_rel.referenced = true;
#endif

    T* join_rel = build_join_relation<T>(left_rel, right_rel);

    join_rel->rows = estimate_join_rows(*join_rel, left_rel, right_rel, info);

    join_rel->cost = compute_join_cost(*join_rel, left_rel, right_rel, info);

    return join_rel;
}

// explicitly instantiate template implementations
// by doing so I avoid defining the template in the header file
template JoinRelation *make_join_relation<JoinRelation>(JoinRelation &, JoinRelation &, GpuqoPlannerInfo*);
template JoinRelationDPE *make_join_relation<JoinRelationDPE>(JoinRelationDPE &, JoinRelationDPE &, GpuqoPlannerInfo*);

template<typename T>
bool do_join(int level, T* &join_rel, T &left_rel, T &right_rel, 
            GpuqoPlannerInfo* info, memo_t &memo, extra_t extra){
    join_rel = make_join_relation<T>(left_rel, right_rel, info);

    auto find_iter = memo.find(join_rel->id);
    if (find_iter != memo.end()){
        JoinRelation* old_jr = find_iter->second;  

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
template bool do_join<JoinRelation>(int, JoinRelation *&, JoinRelation &, JoinRelation &, GpuqoPlannerInfo*, memo_t &, extra_t);
template bool do_join<JoinRelationDPE>(int, JoinRelationDPE *&, JoinRelationDPE &, JoinRelationDPE &, GpuqoPlannerInfo*, memo_t &, extra_t);
