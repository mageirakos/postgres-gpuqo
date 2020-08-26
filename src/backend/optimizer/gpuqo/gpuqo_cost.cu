/*------------------------------------------------------------------------
 *
 * gpuqo_cost.cu
 *      definition of the common cost-computing function
 *
 * src/backend/optimizer/gpuqo/gpuqo_cost.cu
 *
 *-------------------------------------------------------------------------
 */

#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"


__host__ __device__
bool has_useful_index(JoinRelation &left_rel, JoinRelation &right_rel,
                    BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table){
    if (BMS64_SIZE(right_rel.id) != 1)  // inner must be base rel
        return false;
    // -1 since it's 1-indexed, 
    // another -1 since relation with id 0b10 is at index 0 and so on
    int baserel_right_idx = BMS64_LOWEST_POS(right_rel.id) - 2;
    BaseRelation &baserel_right = base_rels[baserel_right_idx];
    RelationID baserel_right_edges = baserel_right.edges;

    while (baserel_right_edges != BMS64_EMPTY){
        int baserel_left_idx = BMS64_LOWEST_POS(baserel_right_edges) - 2;
        if(edge_table[baserel_left_idx*n_rels+baserel_right_idx].has_index)
            return true;
        baserel_right_edges = BMS64_UNSET(baserel_right_edges, baserel_left_idx+1);
    }
    return false;
}

__host__ __device__
double baserel_cost(BaseRelation &base_rel){
    return BASEREL_COEFF * base_rel.tuples;
}


__host__ __device__
double 
compute_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel,
                    BaseRelation* base_rels, int n_rels,
                    EdgeInfo* edge_table)
{
    double hash_cost = HASHJOIN_COEFF * join_rel.rows + left_rel.cost + right_rel.cost;
    double nl_cost = left_rel.cost + left_rel.rows * right_rel.cost;
    double inl_cost;

    if (has_useful_index(left_rel, right_rel, base_rels, n_rels, edge_table)){
        inl_cost = left_rel.cost + INDEXSCAN_COEFF * left_rel.rows * max(join_rel.rows/left_rel.rows, 1.0);
    } else{
        inl_cost = INFD;
    }

    // explicit sort merge
    double sm_cost = (left_rel.cost + right_rel.cost
                        + SORT_COEFF * left_rel.rows * log(left_rel.rows)
                        + SORT_COEFF * right_rel.rows * log(right_rel.rows)
    );

    return min(min(hash_cost, nl_cost), min(inl_cost, sm_cost));
}

__host__ __device__
double 
estimate_join_rows(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, BaseRelation* base_rels, 
                    int n_rels, EdgeInfo* edge_table) 
{
    double sel = 1.0;
    
    // for each baserel of the left relation, for each edge of that base rel
    // getting to the right relation,
    // I multiply the selectivity by the selectivity of that edge
    // NB: edges might be multiple so I need to check every baserel in the left
    // joinrel
    RelationID left_id = left_rel.id;
    while (left_id != BMS64_EMPTY){
        // -1 since it's 1-indexed, 
        // another -1 since relation with id 0b10 is at index 0 and so on
        int baserel_left_idx = BMS64_LOWEST_POS(left_id) - 2;
        BaseRelation &baserel_left = base_rels[baserel_left_idx];
        RelationID baserel_left_edges = baserel_left.edges;

        while (baserel_left_edges != BMS64_EMPTY){
            int baserel_right_idx = BMS64_LOWEST_POS(baserel_left_edges) - 2;
            if (BMS64_IS_SET(right_rel.id, baserel_right_idx+1)){
                sel *= edge_table[baserel_left_idx*n_rels+baserel_right_idx].sel;
            }
            baserel_left_edges = BMS64_UNSET(baserel_left_edges, baserel_right_idx+1);
        }
        left_id = BMS64_UNSET(left_id, baserel_left_idx+1);
    }
    
    double rows = sel * left_rel.rows * right_rel.rows;

    // clamp the number of rows
    return rows > 1 ? round(rows) : 1;
}


__device__
JoinRelation joinCost::operator()(JoinRelation jr){
    JoinRelation left_rel = memo_vals[jr.left_relation_idx];
    JoinRelation right_rel = memo_vals[jr.right_relation_idx];

    jr.rows = estimate_join_rows(jr, left_rel, right_rel,
                                base_rels.get(), n_rels,edge_table.get());

    jr.cost = compute_join_cost(jr, left_rel, right_rel,
                                base_rels.get(), n_rels, edge_table.get());

    return jr;
}
