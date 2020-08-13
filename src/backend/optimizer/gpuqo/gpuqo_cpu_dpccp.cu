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

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"
#include "optimizer/gpuqo_cpu_common.cuh"

typedef void (*emit_f)(RelationID left_id, RelationID right_id,
                    BaseRelation* base_rels, int n_rels,
                    EdgeInfo* edge_table, memo_t &memo, void* extra,
                    struct DPCPUAlgorithm algorithm);

struct GpuqoCPUDPCcpExtra{
    join_f join_function;
};


std::list<RelationID>* get_all_subsets(RelationID set){
    std::list<RelationID> *out = new std::list<RelationID>;
    if (set == BMS64_EMPTY)
        return out;

    RelationID subset = BMS64_LOWEST(set);
    while (subset != set){
        out->push_back(subset);
        subset = BMS64_NEXT_SUBSET(subset, set);
    }
    out->push_back(set);
    return out;
}

RelationID get_neighbours(RelationID set, BaseRelation* base_rels, int n_rels){
    RelationID neigs = BMS64_EMPTY;
    for (int i=1; i <= n_rels; i++){
        RelationID base_rel = BMS64_NTH(i);
        if (BMS64_INTERSECTS(base_rel, set)){
            neigs = BMS64_UNION(neigs, base_rels[i-1].edges);
        }
    }
    return BMS64_DIFFERENCE(neigs, set);
}

void enumerate_csg_rec(RelationID S, RelationID X, RelationID cmp, BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], emit_f emit_function, memo_t &memo, void* extra, struct DPCPUAlgorithm algorithm){
    RelationID N = BMS64_DIFFERENCE(get_neighbours(S, base_rels, n_rels), X);
    std::list<RelationID> *subsets = get_all_subsets(N);
    for (auto subset=subsets->begin(); subset!=subsets->end(); ++subset){
        RelationID emit_set = BMS64_UNION(S, *subset);
        emit_function(cmp, emit_set, base_rels, n_rels, edge_table, memo, extra, algorithm);
    }
    for (auto subset=subsets->begin(); subset!=subsets->end(); ++subset){
        enumerate_csg_rec(BMS64_UNION(S, *subset), BMS64_UNION(X, N), cmp,
            base_rels, n_rels, edge_table, emit_function, memo, extra, algorithm);
    }
    delete subsets; 
}

void enumerate_csg(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], emit_f emit_function, memo_t &memo, void* extra, struct DPCPUAlgorithm algorithm){
    for (int i=n_rels; i>=1; i--){
        RelationID subset = BMS64_NTH(i);

        emit_function(subset, BMS64_EMPTY, base_rels, n_rels, edge_table, memo, extra, algorithm);

        enumerate_csg_rec(subset, BMS64_SET_ALL_LOWER(subset), BMS64_EMPTY,
            base_rels, n_rels, edge_table, emit_function, memo, extra, algorithm);

    }
}

void enumerate_cmp(RelationID S, BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], emit_f emit_function, memo_t &memo, void* extra, struct DPCPUAlgorithm algorithm){
    RelationID X = BMS64_UNION(BMS64_SET_ALL_LOWER(S), S);
    RelationID N = BMS64_DIFFERENCE(get_neighbours(S, base_rels, n_rels), X);
    for (int i=n_rels; i>=1; i--){
        RelationID v = BMS64_NTH(i);
        if (BMS64_INTERSECTS(v, N)){
            emit_function(S, v, base_rels, n_rels, edge_table, memo, extra, algorithm);

            RelationID newX = BMS64_UNION(X, 
                                BMS64_INTERSECTION(BMS64_SET_ALL_LOWER(v), N));
            enumerate_csg_rec(v, newX, S,
                base_rels, n_rels, edge_table, emit_function, memo, extra, algorithm);
        }
    }
}

void gpuqo_cpu_dpccp_init(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], memo_t &memo, void** extra){
    *extra = (void*) new GpuqoCPUDPCcpExtra;
}


void gpuqo_cpu_dpccp_emit(RelationID left_id, RelationID right_id,
                        BaseRelation* base_rels, int n_rels, 
                        EdgeInfo* edge_table, memo_t &memo, void* extra,
                        struct DPCPUAlgorithm algorithm){
    struct GpuqoCPUDPCcpExtra* mExtra = (struct GpuqoCPUDPCcpExtra*) extra;

    if (left_id != BMS64_EMPTY && right_id != BMS64_EMPTY){
        auto left = memo.find(left_id);
        auto right = memo.find(right_id);

        Assert(left != memo.end() && right != memo.end());

        JoinRelation *left_rel = left->second;
        JoinRelation *right_rel = right->second;
        RelationID joinset = BMS64_UNION(left_id, right_id);
        int level = BMS64_SIZE(joinset);

        mExtra->join_function(level, 
            right_id, *right_rel,
            left_id, *left_rel, 
            base_rels, n_rels, edge_table, memo, extra, algorithm
        );

    } else if (left_id != BMS64_EMPTY) {
        enumerate_cmp(left_id, base_rels, n_rels, edge_table, gpuqo_cpu_dpccp_emit, memo, extra, algorithm);
    } else{
        enumerate_cmp(right_id, base_rels, n_rels, edge_table, gpuqo_cpu_dpccp_emit, memo, extra, algorithm);
    }
}

void gpuqo_cpu_dpccp_enumerate(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], join_f join_function, memo_t &memo, void* extra, struct DPCPUAlgorithm algorithm){
    struct GpuqoCPUDPCcpExtra* mExtra = (struct GpuqoCPUDPCcpExtra*) extra;
    mExtra->join_function = join_function;

    enumerate_csg(base_rels, n_rels, edge_table, gpuqo_cpu_dpccp_emit, memo, extra,  algorithm);
}

bool gpuqo_cpu_dpccp_check_join(int level,
                            RelationID left_id, JoinRelation &left_rel,
                            RelationID right_id, JoinRelation &right_rel,
                            BaseRelation* base_rels, int n_rels,
                            EdgeInfo* edge_table, memo_t &memo, void* extra){
    
    // No check is necessary since dpccp guarantees all joinpairs are valid
    Assert(is_disjoint(left_id, right_id) 
        && is_connected(left_id, left_rel, 
                        right_id, right_rel,
                        base_rels, n_rels, edge_table));
    return true;
}

void gpuqo_cpu_dpccp_post_join(int level, bool newrel,
                            RelationID join_id, JoinRelation &join_rel,
                            RelationID left_id, JoinRelation &left_rel,
                            RelationID right_id, JoinRelation &right_rel,
                            BaseRelation* base_rels, int n_rels, 
                            EdgeInfo* edge_table, memo_t &memo, void* extra){
    // nothing to do
}

void gpuqo_cpu_dpccp_teardown(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], memo_t &memo, void* extra){
    delete ((struct GpuqoCPUDPCcpExtra*) extra);
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
extern "C"
QueryTree*
gpuqo_cpu_dpccp(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[])
{
    return gpuqo_cpu_generic(base_rels, n_rels, edge_table, gpuqo_cpu_dpccp_alg);
}
