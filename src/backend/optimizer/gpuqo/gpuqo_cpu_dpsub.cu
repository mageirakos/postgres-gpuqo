/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpsub.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub.cu
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
class DPsubCPUAlgorithm : public CPUAlgorithm<BitmapsetN> {
public:
    virtual void enumerate()
    {
        auto info = CPUAlgorithm<BitmapsetN>::info;
        // first bit is zero
        for (BitmapsetN i=1; i < BitmapsetN::nth(info->n_rels); i++){
            BitmapsetN join_id = i << 1; // first bit is 0 in Postgres

            if (!is_connected(join_id, info->edge_table))
                continue;

            BitmapsetN left_id = join_id.lowest();
            BitmapsetN right_id;
            while (left_id != join_id){
                right_id = join_id - left_id;

                if (!left_id.empty() && !right_id.empty()){
                    auto memo = *CPUAlgorithm<BitmapsetN>::memo;
                    auto left = memo.find(left_id);
                    auto right = memo.find(right_id);

                    if (left != memo.end() && right != memo.end()){
                        JoinRelationCPU<BitmapsetN> *left_rel = left->second;
                        JoinRelationCPU<BitmapsetN> *right_rel = right->second;
                        int level = join_id.size();

                        (*CPUAlgorithm<BitmapsetN>::join)(level, false, *right_rel, *left_rel);
                    }
                }

                left_id = nextSubset(left_id, join_id);
            }
        }
    }

    virtual bool check_join(int level, JoinRelationCPU<BitmapsetN> &left_rel,
        JoinRelationCPU<BitmapsetN> &right_rel)
    {

        // I do not need to check self-connectedness of the single joinrels since 
        // if they were not connected, they wouldn't have been generated and 
        // I would not have been able to find them in the memo
        // I do not need to check if they connect to each other since if they 
        // weren't, join_rel would not be self-connected but I know it is

        Assert(is_connected(left_rel.id, info->edge_table));
        Assert(is_connected(right_rel.id, info->edge_table));
        Assert(are_connected_rel(left_rel, right_rel, 
                                    CPUAlgorithm<BitmapsetN>::info));

        return is_disjoint_rel(left_rel, right_rel);
    }
};

/* gpuqo_cpu_dpsub
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP sub
 *   algorithm.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_dpsub(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPsubCPUAlgorithm<BitmapsetN> alg;
    return gpuqo_cpu_sequential(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_cpu_dpsub<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_dpsub<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);

/* gpuqo_dpe_dpsub
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP sub
 *   algorithm.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_dpe_dpsub(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPsubCPUAlgorithm<BitmapsetN> alg;
    return gpuqo_cpu_dpe(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_dpe_dpsub<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_dpe_dpsub<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
