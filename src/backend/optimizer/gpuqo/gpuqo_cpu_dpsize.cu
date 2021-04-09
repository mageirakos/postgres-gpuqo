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

struct GpuqoCPUDPSizeExtra{
    vector_list_t rels_per_level;

    GpuqoCPUDPSizeExtra(int n_rels) : rels_per_level(n_rels+1) {}
};

void gpuqo_cpu_dpsize_init(GpuqoPlannerInfo* info, memo_t &memo, extra_t &extra){
    extra.alg = (void*) new GpuqoCPUDPSizeExtra(info->n_rels);
    struct GpuqoCPUDPSizeExtra* mExtra = (struct GpuqoCPUDPSizeExtra*) extra.alg;

    for(auto iter = memo.begin(); iter != memo.end(); ++iter){
        mExtra->rels_per_level[1].push_back(iter->second);
    }
}

void gpuqo_cpu_dpsize_enumerate(GpuqoPlannerInfo* info, join_f join_function, memo_t &memo, extra_t extra, struct DPCPUAlgorithm algorithm){
    struct GpuqoCPUDPSizeExtra* mExtra = (struct GpuqoCPUDPSizeExtra*) extra.alg;

    DECLARE_TIMING(iteration);

    for (int join_s=2; join_s<=info->n_rels; join_s++){
        LOG_PROFILE("\nStarting iteration %d\n", join_s);
        START_TIMING(iteration);
        for (int big_s = join_s-1; big_s >= (join_s+1)/2; big_s--){
            int small_s = join_s-big_s;
            for (auto big_i = mExtra->rels_per_level[big_s].begin(); 
                    big_i != mExtra->rels_per_level[big_s].end(); ++big_i){
                for (auto small_i = mExtra->rels_per_level[small_s].begin(); 
                        small_i != mExtra->rels_per_level[small_s].end(); ++small_i){
                    join_function(join_s, true, **big_i, **small_i, 
                        info, memo, extra, algorithm
                    );
                }
            } 
        }
        STOP_TIMING(iteration);
        PRINT_TIMING(iteration);
    }

}

bool gpuqo_cpu_dpsize_check_join(int level, JoinRelationCPU &left_rel,             
                            JoinRelationCPU &right_rel, GpuqoPlannerInfo* info,
                            memo_t &memo, extra_t extra){

    return (is_disjoint(left_rel, right_rel) 
        && are_connected(left_rel, right_rel, info));
}

void gpuqo_cpu_dpsize_post_join(int level, bool newrel, JoinRelationCPU &join_rel, 
                            JoinRelationCPU &left_rel, JoinRelationCPU &right_rel,
                            GpuqoPlannerInfo* info, memo_t &memo, 
                            extra_t extra){
    struct GpuqoCPUDPSizeExtra* mExtra = (struct GpuqoCPUDPSizeExtra*) extra.alg;
    if (newrel)
        mExtra->rels_per_level[level].push_back(&join_rel);
}

void gpuqo_cpu_dpsize_teardown(GpuqoPlannerInfo* info, memo_t &memo, extra_t extra){
    delete ((struct GpuqoCPUDPSizeExtra*) extra.alg);
}

DPCPUAlgorithm gpuqo_cpu_dpsize_alg = {
    .init_function = gpuqo_cpu_dpsize_init,
    .enumerate_function = gpuqo_cpu_dpsize_enumerate,
    .check_join_function = gpuqo_cpu_dpsize_check_join,
    .post_join_function = gpuqo_cpu_dpsize_post_join,
    .teardown_function = gpuqo_cpu_dpsize_teardown
};

/* gpuqo_cpu_dpsize
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
QueryTree*
gpuqo_cpu_dpsize(GpuqoPlannerInfo* info)
{
    return gpuqo_cpu_sequential(info, gpuqo_cpu_dpsize_alg);
}

/* gpuqo_cpu_dpsize
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
QueryTree*
gpuqo_dpe_dpsize(GpuqoPlannerInfo* info)
{
    return gpuqo_cpu_dpe(info, gpuqo_cpu_dpsize_alg);
}


