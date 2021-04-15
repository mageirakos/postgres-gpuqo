/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_sequential.cu
 *      Generic implementation of a sequential CPU algorithm.
 *
 * src/backend/optimizer/gpuqo/gpuqo_cpu_sequential.cu
 *
 *-------------------------------------------------------------------------
 */

#include <list>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_sequential.cuh"

template<typename BitmapsetN>
class SequentialJoinFunction : public CPUJoinFunction<BitmapsetN> {
public:
    SequentialJoinFunction(GpuqoPlannerInfo<BitmapsetN>* _info, 
        memo_t<BitmapsetN>* _memo, CPUAlgorithm<BitmapsetN>* _alg) 
        : CPUJoinFunction<BitmapsetN>(_info, _memo, _alg) {}

    virtual void operator()(int level, bool try_swap,
                JoinRelationCPU<BitmapsetN> &left_rel, 
                JoinRelationCPU<BitmapsetN> &right_rel)
    {
        if (CPUJoinFunction<BitmapsetN>::alg->check_join(level, left_rel, right_rel)){
            JoinRelationCPU<BitmapsetN> *join_rel1, *join_rel2;
            bool new_joinrel;
            new_joinrel = do_join(level, join_rel1, 
                                left_rel, right_rel, CPUJoinFunction<BitmapsetN>::info, *CPUJoinFunction<BitmapsetN>::memo);
            CPUJoinFunction<BitmapsetN>::alg->post_join(level, new_joinrel, *join_rel1, left_rel,  right_rel);
            if (try_swap){
                new_joinrel = do_join(level, join_rel2, right_rel, 
                                    left_rel, CPUJoinFunction<BitmapsetN>::info, *CPUJoinFunction<BitmapsetN>::memo);
                CPUJoinFunction<BitmapsetN>::alg->post_join(level, new_joinrel, *join_rel2, left_rel, right_rel);
            }
        }
    }
};

template<typename BitmapsetN>
QueryTree<BitmapsetN>* gpuqo_cpu_sequential(GpuqoPlannerInfo<BitmapsetN>* info, 
                                CPUAlgorithm<BitmapsetN> *algorithm)
{
    
    DECLARE_TIMING(gpuqo_cpu_sequential);
    START_TIMING(gpuqo_cpu_sequential);

    memo_t<BitmapsetN> memo;
    QueryTree<BitmapsetN>* out = NULL;

    for(int i=0; i<info->n_rels; i++){
        JoinRelationCPU<BitmapsetN> *jr = new JoinRelationCPU<BitmapsetN>;
        jr->id = info->base_rels[i].id; 
        jr->left_rel_id = 0; 
        jr->left_rel_ptr = NULL; 
        jr->right_rel_id = 0; 
        jr->right_rel_ptr = NULL; 
        jr->cost = baserel_cost(info->base_rels[i]); 
        jr->rows = info->base_rels[i].rows; 
        jr->edges = info->edge_table[i];
        memo.insert(std::make_pair(info->base_rels[i].id, jr));
    }

    SequentialJoinFunction<BitmapsetN> join_func(info, &memo, algorithm);

    algorithm->init(info, &memo, &join_func);
    
    algorithm->enumerate();

    BitmapsetN final_joinrel_id = BitmapsetN(0);
    for (int i = 0; i < info->n_rels; i++)
        final_joinrel_id |= info->base_rels[i].id;

    
    auto final_joinrel_pair = memo.find(final_joinrel_id);
    if (final_joinrel_pair != memo.end())
        build_query_tree(final_joinrel_pair->second, memo, &out);

    // delete all dynamically allocated memory
    for (auto iter=memo.begin(); iter != memo.end(); ++iter){
        delete iter->second;
    }

    STOP_TIMING(gpuqo_cpu_sequential);
    PRINT_TIMING(gpuqo_cpu_sequential);

    return out;
}

template QueryTree<Bitmapset32>* gpuqo_cpu_sequential<Bitmapset32>
        (GpuqoPlannerInfo<Bitmapset32>*, CPUAlgorithm<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_sequential<Bitmapset64>
        (GpuqoPlannerInfo<Bitmapset64>*, CPUAlgorithm<Bitmapset64>*);
