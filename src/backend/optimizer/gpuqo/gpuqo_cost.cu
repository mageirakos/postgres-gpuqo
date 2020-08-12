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
double 
compute_join_cost(JoinRelation &join_rel, 
                    RelationID &left_id, JoinRelation &left_rel,
                    RelationID &right_id, JoinRelation &right_rel,
                    BaseRelation* base_rels, EdgeInfo* edge_table,
                    int number_of_rels
                )
{
    // this cost function represents the "cost" of an hash join
    // once again, this is pretty random
    return join_rel.rows + left_rel.cost + right_rel.cost;
}

__host__ __device__
double 
estimate_join_rows(JoinRelation &join_rel, 
                    RelationID &left_id, JoinRelation &left_rel,
                    RelationID &right_id, JoinRelation &right_rel,
                    BaseRelation* base_rels, EdgeInfo* edge_table,
                    int number_of_rels
                ) 
{
    double sel = 1.0;
    
    // for each baserel of the left relation, for each edge of that base rel
    // getting to the right relation,
    // I multiply the selectivity by the selectivity of that edge
    // NB: edges might be multiple so I need to check every baserel in the left
    // joinrel
    for (int i = 1; i <= number_of_rels; i++){
        uint64_t base_relid_left = 1<<i;
        BaseRelation baserel_left = base_rels[i-1];
        if (base_relid_left & left_id){
            for (int j = 1; j <= number_of_rels; j++){
                uint64_t base_relid_right = 1<<j;
                if (baserel_left.edges & right_id & base_relid_right){
                    sel *= edge_table[(i-1)*number_of_rels+(j-1)].sel;
                }
            }
        }
    }
    
    double rows = sel * left_rel.rows * right_rel.rows;

    // clamp the number of rows
    return rows > 1 ? round(rows) : 1;
}


__device__
JoinRelation joinCost::operator()(JoinRelation jr){
    RelationID left_id = memo_keys[jr.left_relation_idx];
    RelationID right_id = memo_keys[jr.right_relation_idx];
    JoinRelation left_rel = memo_vals[jr.left_relation_idx];
    JoinRelation right_rel = memo_vals[jr.right_relation_idx];

    jr.rows = estimate_join_rows(jr, left_id, left_rel, right_id, right_rel,
                                base_rels.get(), edge_table.get(), n_rels);

    jr.cost = compute_join_cost(jr, left_id, left_rel, right_id, right_rel,
                                base_rels.get(), edge_table.get(), n_rels);

    return jr;
}
