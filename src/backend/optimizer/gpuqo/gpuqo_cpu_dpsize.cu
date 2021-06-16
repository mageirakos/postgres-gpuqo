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

template<typename BitmapsetN, typename memo_t>
class DPsizeCPUAlgorithm : public CPUAlgorithm<BitmapsetN, memo_t> {
private:

    vector_list_t<BitmapsetN> rels_per_level;

public:

    virtual void init(GpuqoPlannerInfo<BitmapsetN>* _info, 
        memo_t* _memo,
        CPUJoinFunction<BitmapsetN, memo_t> *_join)
    {
        CPUAlgorithm<BitmapsetN, memo_t>::init(_info, _memo, _join);

        rels_per_level = vector_list_t<BitmapsetN>(CPUAlgorithm<BitmapsetN, memo_t>::info->n_rels+1);

        auto &memo = *CPUAlgorithm<BitmapsetN, memo_t>::memo;

        for(auto iter = memo.begin(); iter != memo.end(); ++iter){
            rels_per_level[1].push_back(iter->second);
        }
    }

    virtual void enumerate(){
        DECLARE_TIMING(iteration);

        GpuqoPlannerInfo<BitmapsetN>* info = CPUAlgorithm<BitmapsetN, memo_t>::info;

        for (int join_s=2; join_s<=info->n_iters; join_s++){
            LOG_PROFILE("\nStarting iteration %d\n", join_s);
            START_TIMING(iteration);
            for (int left_s = join_s-1; left_s >= 1; left_s--){
                int right_s = join_s-left_s;
                for (auto &left_i : rels_per_level[left_s]){
                    for (auto &right_i : rels_per_level[right_s]){
#ifdef GPUQO_PRINT_N_JOINS
                        CPUAlgorithm<BitmapsetN, memo_t>::n_checks++;
#endif
                        (*CPUAlgorithm<BitmapsetN, memo_t>::join)(join_s, false, *left_i, *right_i);
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
            && are_connected_rel(left_rel, right_rel, CPUAlgorithm<BitmapsetN, memo_t>::info));
    }

    virtual void post_join(int level, bool newrel, 
                JoinRelationCPU<BitmapsetN> &join_rel, 
                JoinRelationCPU<BitmapsetN> &left_rel, 
                JoinRelationCPU<BitmapsetN> &right_rel)
    {
        CPUAlgorithm<BitmapsetN, memo_t>::post_join(level, newrel, 
                                            join_rel, left_rel, right_rel);
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
    DPsizeCPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN> > alg;
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
    DPsizeCPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN> > alg;
    return gpuqo_cpu_dpe(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_dpe_dpsize<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_dpe_dpsize<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);

