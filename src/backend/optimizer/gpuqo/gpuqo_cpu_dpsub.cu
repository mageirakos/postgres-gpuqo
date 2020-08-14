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

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"
#include "optimizer/gpuqo_cpu_common.cuh"

void gpuqo_cpu_dpsub_init(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], memo_t &memo, void** extra){
    // nothing to do
}

void gpuqo_cpu_dpsub_enumerate(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], join_f join_function, memo_t &memo, void* extra, struct DPCPUAlgorithm algorithm){

    // first bit is zero
    for (RelationID i=1; i < BMS64_NTH(n_rels); i++){
        RelationID join_id = i << 1; // first bit is 0 in Postgres
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

                    join_function(level, false, *right_rel, *left_rel, 
                        base_rels, n_rels, edge_table, memo, extra, algorithm
                    );
                }
            }

            left_id = BMS64_NEXT_SUBSET(left_id, join_id);
        }
    }
}

bool gpuqo_cpu_dpsub_check_join(int level, JoinRelation &left_rel,
                            JoinRelation &right_rel, BaseRelation* base_rels, 
                            int n_rels, EdgeInfo* edge_table, memo_t &memo, 
                            void* extra){

    // I do not need to check connectedness of the single joinrels since 
    // if they were not connected, they wouldn't have been generated and 
    // I would not have been able to find them in the memo
    return (is_disjoint(left_rel, right_rel) 
        && are_connected(left_rel, right_rel,
                        base_rels, n_rels, edge_table));
}

void gpuqo_cpu_dpsub_post_join(int level, bool newrel, JoinRelation &join_rel, 
                            JoinRelation &left_rel, JoinRelation &right_rel,
                            BaseRelation* base_rels, int n_rels, 
                            EdgeInfo* edge_table, memo_t &memo, void* extra){
    // nothing to do
}

void gpuqo_cpu_dpsub_teardown(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], memo_t &memo, void* extra){
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
gpuqo_cpu_dpsub(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[])
{
    return gpuqo_cpu_generic(base_rels, n_rels, edge_table, gpuqo_cpu_dpsub_alg);
}


