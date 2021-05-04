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
#include "gpuqo_cpu_common.cuh"
#include "gpuqo_cpu_dpsub.cuh"
#include "gpuqo_cpu_sequential.cuh"
#include "gpuqo_cpu_dpe.cuh"

template<typename BitmapsetN, typename memo_t, bool manage_best>
class DPsubCPUAlgorithm : public DPsubGenericCPUAlgorithm<BitmapsetN, memo_t, manage_best> {
public:
    virtual JoinRelationCPU<BitmapsetN> *enumerate_subsets(BitmapsetN join_id){
        BitmapsetN left_id = join_id.lowest();
        BitmapsetN right_id;

        JoinRelationCPU<BitmapsetN> *join_rel = NULL;

        while (left_id != join_id){
            right_id = join_id - left_id;

            if (!left_id.empty() && !right_id.empty()){
                auto &memo = *CPUAlgorithm<BitmapsetN, memo_t>::memo;
                auto left = memo.find(left_id);
                auto right = memo.find(right_id);

                if (left != memo.end() && right != memo.end()){
                    JoinRelationCPU<BitmapsetN> *left_rel = left->second;
                    JoinRelationCPU<BitmapsetN> *right_rel = right->second;
                    int level = join_id.size();

                    JoinRelationCPU<BitmapsetN> *new_join_rel = 
                        (*CPUAlgorithm<BitmapsetN, memo_t>::join)(
                                level, false, *left_rel, *right_rel);

                    if (manage_best){
                        if (join_rel == NULL 
                                || (new_join_rel != NULL 
                                    && new_join_rel->cost < join_rel->cost))
                        {
                            if (join_rel != NULL)
                                delete join_rel;

                            join_rel = new_join_rel;
                        } else {
                            if (new_join_rel != NULL)
                                delete new_join_rel;
                        }
                    }
                }
            }

            left_id = nextSubset(left_id, join_id);
        }

        return join_rel;
    }

    virtual bool check_join(int level, JoinRelationCPU<BitmapsetN> &left_rel,
        JoinRelationCPU<BitmapsetN> &right_rel)
    {

        // I do not need to check self-connectedness of the single joinrels since 
        // if they were not connected, they wouldn't have been generated and 
        // I would not have been able to find them in the memo
        // I do not need to check if they connect to each other since if they 
        // weren't, join_rel would not be self-connected but I know it is

#ifdef USE_ASSERT_CHECKING
        auto &info = CPUAlgorithm<BitmapsetN, memo_t>::info;
#endif

        Assert(is_connected(left_rel.id, info->edge_table));
        Assert(is_connected(right_rel.id, info->edge_table));
        Assert(are_connected_rel(left_rel, right_rel, info));

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
    DPsubCPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN>,false> alg;
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
    DPsubCPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN>,false> alg;
    return gpuqo_cpu_dpe(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_dpe_dpsub<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_dpe_dpsub<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);

/* gpuqo_cpu_dpsub_parallel
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP sub
 *   algorithm with BiCC optimization.
 */
 template<typename BitmapsetN>
 QueryTree<BitmapsetN>*
 gpuqo_cpu_dpsub_parallel(GpuqoPlannerInfo<BitmapsetN>* info)
 {
    DPsubCPUAlgorithm<BitmapsetN, level_hashtable<BitmapsetN>, true> alg;
    return gpuqo_cpu_dpsub_generic_parallel(info, &alg);
 }
 
 template QueryTree<Bitmapset32>* gpuqo_cpu_dpsub_parallel<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
 template QueryTree<Bitmapset64>* gpuqo_cpu_dpsub_parallel<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
 