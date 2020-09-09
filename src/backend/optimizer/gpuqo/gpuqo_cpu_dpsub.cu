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

void gpuqo_cpu_dpsub_init(GpuqoPlannerInfo* info, memo_t &memo, extra_t &extra){
    // nothing to do
}

void gpuqo_cpu_dpsub_enumerate(GpuqoPlannerInfo* info, join_f join_function, memo_t &memo, extra_t extra, struct DPCPUAlgorithm algorithm){

    // first bit is zero
    for (RelationID i=1; i < BMS64_NTH(info->n_rels); i++){
        RelationID join_id = i << 1; // first bit is 0 in Postgres

        if (!is_connected(join_id, info))
            continue;

        RelationID left_id = BMS64_LOWEST(join_id);
        RelationID right_id;
        while (left_id != join_id){
            right_id = BMS64_DIFFERENCE(join_id, left_id);

            if (left_id != BMS64_EMPTY && right_id != BMS64_EMPTY){
                auto left = memo.find(left_id);
                auto right = memo.find(right_id);

                if (left != memo.end() && right != memo.end()){
                    JoinRelation *left_rel = left->second;
                    JoinRelation *right_rel = right->second;
                    int level = BMS64_SIZE(join_id);

                    join_function(level, false, *right_rel, *left_rel, info,    
                            memo, extra, algorithm
                    );
                }
            }

            left_id = BMS64_NEXT_SUBSET(left_id, join_id);
        }
    }
}

bool gpuqo_cpu_dpsub_check_join(int level, JoinRelation &left_rel,
                            JoinRelation &right_rel, GpuqoPlannerInfo* info, 
                            memo_t &memo, extra_t extra){

    // I do not need to check connectedness of the single joinrels since 
    // if they were not connected, they wouldn't have been generated and 
    // I would not have been able to find them in the memo
    return (is_disjoint(left_rel, right_rel) 
        && are_connected(left_rel, right_rel, info));
}

void gpuqo_cpu_dpsub_post_join(int level, bool newrel, JoinRelation &join_rel, 
                            JoinRelation &left_rel, JoinRelation &right_rel,
                            GpuqoPlannerInfo* info, memo_t &memo, 
                            extra_t extra){
    // nothing to do
}

void gpuqo_cpu_dpsub_teardown(GpuqoPlannerInfo* info, memo_t &memo, extra_t extra){
    // nothing to do
}

DPCPUAlgorithm gpuqo_cpu_dpsub_alg = {
    .init_function = gpuqo_cpu_dpsub_init,
    .enumerate_function = gpuqo_cpu_dpsub_enumerate,
    .check_join_function = gpuqo_cpu_dpsub_check_join,
    .post_join_function = gpuqo_cpu_dpsub_post_join,
    .teardown_function = gpuqo_cpu_dpsub_teardown
};

/* gpuqo_cpu_dpsub
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP sub
 *   algorithm.
 */
extern "C"
QueryTree*
gpuqo_cpu_dpsub(GpuqoPlannerInfo* info)
{
    return gpuqo_cpu_sequential(info, gpuqo_cpu_dpsub_alg);
}


/* gpuqo_dpe_dpsub
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP sub
 *   algorithm.
 */
extern "C"
QueryTree*
gpuqo_dpe_dpsub(GpuqoPlannerInfo* info)
{
    return gpuqo_cpu_dpe(info, gpuqo_cpu_dpsub_alg);
}


