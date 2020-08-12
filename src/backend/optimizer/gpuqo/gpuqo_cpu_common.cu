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

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"
#include "optimizer/gpuqo_cpu_common.cuh"

void build_query_tree(RelationID relid, memo_t &memo, QueryTree **qt)
{
    auto relid_jr_pair = memo.find(relid);
    if (relid_jr_pair == memo.end()){
        (*qt) = NULL;
        return;
    }
    JoinRelation* jr = relid_jr_pair->second;

    (*qt) = new QueryTree;
    (*qt)->id = relid;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr->rows;
    (*qt)->cost = jr->cost;

    build_query_tree(jr->left_relation_id, memo, &((*qt)->left));
    build_query_tree(jr->right_relation_id, memo, &((*qt)->right));
}

/* make_join_relation
 *
 *	 Builds a join relation from left and right relations.
 */
JoinRelation* make_join_relation(RelationID left_id, JoinRelation &left_rel,
                                 RelationID right_id, JoinRelation &right_rel,
                                 BaseRelation* base_rels, EdgeInfo* edge_table,
                                 int number_of_rels){

    JoinRelation* join_rel = new JoinRelation;

    join_rel->left_relation_id = left_id;
    join_rel->right_relation_id = right_id;
    join_rel->edges = left_rel.edges | right_rel.edges;

    join_rel->rows = estimate_join_rows(
        *join_rel,
        left_id, left_rel,
        right_id, right_rel,
        base_rels, edge_table, number_of_rels
    );

    join_rel->cost = compute_join_cost(
        *join_rel,
        left_id, left_rel,
        right_id, right_rel,
        base_rels, edge_table, number_of_rels
    );

    return join_rel;
}

bool do_join(int level, 
            RelationID &join_id, JoinRelation* &join_rel,
            RelationID left_id, JoinRelation &left_rel,
            RelationID right_id, JoinRelation &right_rel,
            BaseRelation* base_rels, EdgeInfo* edge_table,
            int number_of_rels, memo_t &memo, void* extra){
    join_id = left_id | right_id;
    join_rel = make_join_relation(
        left_id, left_rel,
        right_id, right_rel,
        base_rels, edge_table, number_of_rels
    );

    auto find_iter = memo.find(join_id);
    if (find_iter != memo.end()){
        JoinRelation* old_jr = find_iter->second;
        if (join_rel->cost < old_jr->cost){
            *old_jr = *join_rel;
        }
        delete join_rel;
        return false;
    } else{
        memo.insert(std::make_pair(join_id, join_rel));
        return true;
    }
}

void gpuqo_cpu_generic_join(int level, 
                            RelationID left_id, JoinRelation &left_rel,
                            RelationID right_id, JoinRelation &right_rel,
                            BaseRelation* base_rels, EdgeInfo* edge_table,
                            int number_of_rels, memo_t &memo, void* extra, 
                            struct DPCPUAlgorithm algorithm){
    if (algorithm.check_join_function(level, left_id, left_rel, 
                            right_id, right_rel,
                            base_rels, edge_table, number_of_rels, memo, extra)){
        RelationID join_id1, join_id2;
        JoinRelation *join_rel1, *join_rel2;
        bool new_joinrel;
        new_joinrel = do_join(level, 
                            join_id1, join_rel1,
                            left_id, left_rel, 
                            right_id, right_rel,
                            base_rels, edge_table, number_of_rels, memo, extra);
        algorithm.post_join_function(level, new_joinrel, 
                            join_id1, *join_rel1,
                            left_id, left_rel, 
                            right_id, right_rel,
                            base_rels, edge_table, number_of_rels, memo, extra);
        new_joinrel = do_join(level, 
                            join_id2, join_rel2,
                            right_id, right_rel,
                            left_id, left_rel, 
                            base_rels, edge_table, number_of_rels, memo, extra);
        algorithm.post_join_function(level, new_joinrel, 
                            join_id2, *join_rel2,
                            left_id, left_rel, 
                            right_id, right_rel,
                            base_rels, edge_table, number_of_rels, memo, extra);
    }
}

QueryTree* gpuqo_cpu_generic(BaseRelation baserels[], int N, 
                             EdgeInfo edge_table[], DPCPUAlgorithm algorithm){
    
    DECLARE_TIMING(gpuqo_cpu_dpsize);
    START_TIMING(gpuqo_cpu_dpsize);

    void* extra;
    memo_t memo;
    QueryTree* out = NULL;
    RelationID joinrel = 0ULL;

    for(int i=0; i<N; i++){
        JoinRelation *jr = new JoinRelation;
        jr->left_relation_id = 0; 
        jr->right_relation_id = 0; 
        jr->cost = 0.2*baserels[i].rows; 
        jr->rows = baserels[i].rows; 
        jr->edges = baserels[i].edges;
        memo.insert(std::make_pair(baserels[i].id, jr));
    }

    algorithm.init_function(baserels, N, edge_table, memo, &extra);
    
    algorithm.enumerate_function(baserels, N, edge_table,gpuqo_cpu_generic_join, memo, extra, algorithm);

    for (int i = 0; i < N; i++)
        joinrel |= baserels[i].id;

    build_query_tree(joinrel, memo, &out);

    // delete all dynamically allocated memory
    for (auto iter=memo.begin(); iter != memo.end(); ++iter){
        delete iter->second;
    }

    algorithm.teardown_function(baserels, N, edge_table, memo, extra);

    STOP_TIMING(gpuqo_cpu_dpsize);
    PRINT_TIMING(gpuqo_cpu_dpsize);

    return out;
}