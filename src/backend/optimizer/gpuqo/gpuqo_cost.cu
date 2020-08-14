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
compute_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel,
                    BaseRelation* base_rels, int n_rels,
                    EdgeInfo* edge_table)
{
    // this cost function represents the "cost" of an hash join
    // once again, this is pretty random
    return join_rel.rows + left_rel.cost + right_rel.cost;
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
    for (int i = 1; i <= n_rels; i++){
        RelationID base_relid_left = BMS64_NTH(i);
        BaseRelation baserel_left = base_rels[i-1];
        if (BMS64_INTERSECTS(base_relid_left, left_rel.id)){
            for (int j = 1; j <= n_rels; j++){
                RelationID base_relid_right = BMS64_NTH(j);
                if (BMS64_INTERSECTS(BMS64_INTERSECTION(baserel_left.edges, right_rel.id), base_relid_right)){
                    sel *= edge_table[(i-1)*n_rels+(j-1)].sel;
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
    JoinRelation left_rel = memo_vals[jr.left_relation_idx];
    JoinRelation right_rel = memo_vals[jr.right_relation_idx];

    jr.rows = estimate_join_rows(jr, left_rel, right_rel,
                                base_rels.get(), n_rels,edge_table.get());

    jr.cost = compute_join_cost(jr, left_rel, right_rel,
                                base_rels.get(), n_rels, edge_table.get());

    return jr;
}
