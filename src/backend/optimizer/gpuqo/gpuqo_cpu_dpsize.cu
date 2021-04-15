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

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_sequential.cuh"
#include "gpuqo_cpu_dpe.cuh"

template<typename BitmapsetN>
class DPsizeCPUAlgorithm : public CPUAlgorithm<BitmapsetN> {
private:

    vector_list_t<BitmapsetN> rels_per_level;

public:

    virtual void init(GpuqoPlannerInfo<BitmapsetN>* _info, 
        memo_t<BitmapsetN>* _memo,
        CPUJoinFunction<BitmapsetN> *_join)
    {
        CPUAlgorithm<BitmapsetN>::init(_info, _memo, _join);

        rels_per_level = vector_list_t<BitmapsetN>(CPUAlgorithm<BitmapsetN>::info->n_rels+1);

        auto &memo = *CPUAlgorithm<BitmapsetN>::memo;

        for(auto iter = memo.begin(); iter != memo.end(); ++iter){
            rels_per_level[1].push_back(iter->second);
        }
    }

    virtual void enumerate(){
        DECLARE_TIMING(iteration);

        GpuqoPlannerInfo<BitmapsetN>* info = CPUAlgorithm<BitmapsetN>::info;

        for (int join_s=2; join_s<=info->n_rels; join_s++){
            LOG_PROFILE("\nStarting iteration %d\n", join_s);
            START_TIMING(iteration);
            for (int big_s = join_s-1; big_s >= (join_s+1)/2; big_s--){
                int small_s = join_s-big_s;
                for (auto big_i = rels_per_level[big_s].begin(); 
                        big_i != rels_per_level[big_s].end(); ++big_i){
                    for (auto small_i = rels_per_level[small_s].begin(); 
                             small_i != rels_per_level[small_s].end(); ++small_i){
                        (*CPUAlgorithm<BitmapsetN>::join)(join_s, true, **big_i, **small_i);
                    }
                } 
            }
            STOP_TIMING(iteration);
            PRINT_TIMING(iteration);
        }

    }

    virtual bool check_join(int level, 
                    JoinRelationCPU<BitmapsetN> &left_rel,             
                    JoinRelationCPU<BitmapsetN> &right_rel)
    {
        return (is_disjoint_rel(left_rel, right_rel) 
            && are_connected_rel(left_rel, right_rel, CPUAlgorithm<BitmapsetN>::info));
    }

    virtual void post_join(int level, bool newrel, 
                JoinRelationCPU<BitmapsetN> &join_rel, 
                JoinRelationCPU<BitmapsetN> &left_rel, 
                JoinRelationCPU<BitmapsetN> &right_rel)
    {
        if (newrel)
            rels_per_level[level].push_back(&join_rel);
    }
};

/* gpuqo_cpu_dpsize
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_dpsize(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPsizeCPUAlgorithm<BitmapsetN> alg;
    return gpuqo_cpu_sequential(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_cpu_dpsize<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_dpsize<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);

/* gpuqo_cpu_dpsize
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_dpe_dpsize(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPsizeCPUAlgorithm<BitmapsetN> alg;
    return gpuqo_cpu_dpe(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_dpe_dpsize<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_dpe_dpsize<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);

