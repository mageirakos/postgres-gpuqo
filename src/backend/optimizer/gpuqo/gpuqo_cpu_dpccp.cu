/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpccp.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpccp.cu
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

typedef void (*emit_f)(RelationID left_id, RelationID right_id,
                    GpuqoPlannerInfo* info, memo_t &memo, extra_t extra,
                    struct DPCPUAlgorithm algorithm);

struct GpuqoCPUDPCcpExtra{
    join_f join_function;
};


std::list<RelationID>* get_all_subsets(RelationID set){
    std::list<RelationID> *out = new std::list<RelationID>;
    if (set.empty())
        return out;

    RelationID subset = set.lowest();
    while (subset != set){
        out->push_back(subset);
        subset = nextSubset(subset, set);
    }
    out->push_back(set);
    return out;
}

void enumerate_csg_rec(RelationID S, RelationID X, RelationID cmp, 
                    GpuqoPlannerInfo* info, emit_f emit_function, memo_t &memo,
                    extra_t extra, struct DPCPUAlgorithm algorithm){
    LOG_DEBUG("enumerate_csg_rec(%u, %u, %u)\n", S.toUint(), X.toUint(), cmp.toUint());
    RelationID N = get_neighbours(S, info->edge_table) - X;
    std::list<RelationID> *subsets = get_all_subsets(N);
    for (auto subset=subsets->begin(); subset!=subsets->end(); ++subset){
        RelationID emit_set = S | *subset;
        emit_function(cmp, emit_set, info, memo, extra, algorithm);
    }
    for (auto subset=subsets->begin(); subset!=subsets->end(); ++subset){
        enumerate_csg_rec(S|*subset, X|N, cmp,
            info, emit_function, memo, extra, algorithm);
    }
    delete subsets; 
}

void enumerate_csg(GpuqoPlannerInfo* info, emit_f emit_function, memo_t &memo, extra_t extra, struct DPCPUAlgorithm algorithm){
    for (int i=info->n_rels; i>=1; i--){
        RelationID subset = RelationID::nth(i);

        emit_function(subset, RelationID(0), info, memo, extra, algorithm);

        enumerate_csg_rec(subset, subset.allLowerInc(), RelationID(0),
            info, emit_function, memo, extra, algorithm);

    }
}

void enumerate_cmp(RelationID S, GpuqoPlannerInfo* info, emit_f emit_function, memo_t &memo, extra_t extra, struct DPCPUAlgorithm algorithm){
    LOG_DEBUG("enumerate_cmp(%u)\n", S.toUint());
    RelationID X = S.allLowerInc();
    RelationID N = get_neighbours(S, info->edge_table) - X;
    RelationID temp = N;
    while (!temp.empty()){
        int idx = temp.highestPos();
        RelationID v = RelationID::nth(idx);
        emit_function(S, v, info, memo, extra, algorithm);

        RelationID newX = X | (v.allLowerInc() & N);
        enumerate_csg_rec(v, newX, S, info, emit_function, memo, extra, 
                        algorithm);
        
        temp -= v;
    }
}

void gpuqo_cpu_dpccp_init(GpuqoPlannerInfo* info, memo_t &memo, extra_t &extra){
    extra.alg = (void*) new GpuqoCPUDPCcpExtra;
}


void gpuqo_cpu_dpccp_emit(RelationID left_id, RelationID right_id,
                        GpuqoPlannerInfo* info, memo_t &memo, 
                        extra_t extra, struct DPCPUAlgorithm algorithm){
    LOG_DEBUG("gpuqo_cpu_dpccp_emit(%u, %u)\n", left_id.toUint(), right_id.toUint());

    struct GpuqoCPUDPCcpExtra* mExtra = (struct GpuqoCPUDPCcpExtra*) extra.alg;

    if (!left_id.empty() && !right_id.empty()){
        auto left = memo.find(left_id);
        auto right = memo.find(right_id);

        Assert(left != memo.end() && right != memo.end());

        JoinRelationCPU *left_rel = left->second;
        JoinRelationCPU *right_rel = right->second;
        RelationID joinset = left_id | right_id;
        int level = joinset.size();

        mExtra->join_function(level, true, *right_rel, *left_rel, info, 
                memo, extra, algorithm
        );

    } else if (!left_id.empty()) {
        enumerate_cmp(left_id, info, gpuqo_cpu_dpccp_emit, 
                memo, extra, algorithm);
    } else{
        enumerate_cmp(right_id, info, gpuqo_cpu_dpccp_emit, 
                memo, extra, algorithm);
    }
}

void gpuqo_cpu_dpccp_enumerate(GpuqoPlannerInfo* info, join_f join_function, memo_t &memo, extra_t extra, struct DPCPUAlgorithm algorithm){
    struct GpuqoCPUDPCcpExtra* mExtra = (struct GpuqoCPUDPCcpExtra*) extra.alg;
    mExtra->join_function = join_function;

    enumerate_csg(info, gpuqo_cpu_dpccp_emit, memo, extra,  algorithm);
}

bool gpuqo_cpu_dpccp_check_join(int level, JoinRelationCPU &left_rel,
                            JoinRelationCPU &right_rel, GpuqoPlannerInfo* info, 
                            memo_t &memo, extra_t extra){
    
    // No check is necessary since dpccp guarantees all joinpairs are valid
    Assert(is_disjoint(left_rel, right_rel) 
        && are_connected(left_rel, right_rel, info));
    return true;
}

void gpuqo_cpu_dpccp_post_join(int level, bool newrel, JoinRelationCPU &join_rel, 
                            JoinRelationCPU &left_rel, JoinRelationCPU &right_rel,
                            GpuqoPlannerInfo* info, 
                            memo_t &memo, extra_t extra){
    // nothing to do
}

void gpuqo_cpu_dpccp_teardown(GpuqoPlannerInfo* info, memo_t &memo, extra_t extra){
    delete ((struct GpuqoCPUDPCcpExtra*) extra.alg);
}

DPCPUAlgorithm gpuqo_cpu_dpccp_alg = {
    .init_function = gpuqo_cpu_dpccp_init,
    .enumerate_function = gpuqo_cpu_dpccp_enumerate,
    .check_join_function = gpuqo_cpu_dpccp_check_join,
    .post_join_function = gpuqo_cpu_dpccp_post_join,
    .teardown_function = gpuqo_cpu_dpccp_teardown
};

/* gpuqo_cpu_dpccp
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
QueryTree*
gpuqo_cpu_dpccp(GpuqoPlannerInfo* info)
{
    return gpuqo_cpu_sequential(info, gpuqo_cpu_dpccp_alg);
}

/* gpuqo_dpe_dpccp
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
QueryTree*
gpuqo_dpe_dpccp(GpuqoPlannerInfo* info)
{
    return gpuqo_cpu_dpe(info, gpuqo_cpu_dpccp_alg);
}
