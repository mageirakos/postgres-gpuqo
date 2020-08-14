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

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"
#include "optimizer/gpuqo_cpu_sequential.cuh"

void gpuqo_cpu_sequential_join(int level, bool try_swap,
                            JoinRelation &left_rel, JoinRelation &right_rel,
                            BaseRelation* base_rels, int n_rels, 
                            EdgeInfo* edge_table, memo_t &memo, extra_t extra, 
                            struct DPCPUAlgorithm algorithm){
    if (algorithm.check_join_function(level, left_rel, right_rel,
                            base_rels, n_rels, edge_table, memo, extra)){
        JoinRelation *join_rel1, *join_rel2;
        bool new_joinrel;
        new_joinrel = do_join<JoinRelation>(level, join_rel1, 
                            left_rel, right_rel, base_rels, n_rels, edge_table, 
                            memo, extra);
        algorithm.post_join_function(level, new_joinrel, *join_rel1, 
                            left_rel,  right_rel, base_rels, n_rels,
                            edge_table, memo, extra);
        if (try_swap){
            new_joinrel = do_join<JoinRelation>(level, join_rel2, right_rel, 
                                left_rel, base_rels, n_rels, edge_table, memo,
                                extra);
            algorithm.post_join_function(level, new_joinrel, *join_rel2,
                                left_rel, right_rel, base_rels, n_rels,
                                edge_table, memo, extra);
        }
    }
}

QueryTree* gpuqo_cpu_sequential(BaseRelation base_rels[], int n_rels, 
                             EdgeInfo edge_table[], DPCPUAlgorithm algorithm){
    
    DECLARE_TIMING(gpuqo_cpu_sequential);
    START_TIMING(gpuqo_cpu_sequential);

    extra_t extra;
    memo_t memo;
    QueryTree* out = NULL;

    for(int i=0; i<n_rels; i++){
        JoinRelation *jr = new JoinRelation;
        jr->id = base_rels[i].id; 
        jr->left_relation_id = 0; 
        jr->left_relation_ptr = NULL; 
        jr->right_relation_id = 0; 
        jr->right_relation_ptr = NULL; 
        jr->cost = 0.2*base_rels[i].rows; 
        jr->rows = base_rels[i].rows; 
        jr->edges = base_rels[i].edges;
        memo.insert(std::make_pair(base_rels[i].id, jr));
    }

    algorithm.init_function(base_rels, n_rels, edge_table, memo, extra);
    
    algorithm.enumerate_function(base_rels, n_rels, edge_table,gpuqo_cpu_sequential_join, memo, extra, algorithm);

    RelationID final_joinrel_id = 0ULL;
    for (int i = 0; i < n_rels; i++)
        final_joinrel_id = BMS64_UNION(final_joinrel_id, base_rels[i].id);

    
    auto final_joinrel_pair = memo.find(final_joinrel_id);
    if (final_joinrel_pair != memo.end())
        build_query_tree(final_joinrel_pair->second, memo, &out);

    // delete all dynamically allocated memory
    for (auto iter=memo.begin(); iter != memo.end(); ++iter){
        delete iter->second;
    }

    algorithm.teardown_function(base_rels, n_rels, edge_table, memo, extra);

    STOP_TIMING(gpuqo_cpu_sequential);
    PRINT_TIMING(gpuqo_cpu_sequential);

    return out;
}