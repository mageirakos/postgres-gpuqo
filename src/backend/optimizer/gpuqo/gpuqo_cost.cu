/*------------------------------------------------------------------------
 *
 * gpuqo_cost.c
 *      definition of the common cost-computing function
 *
 * src/backend/optimizer/gpuqo/gpuqo_cost.c
 *
 *-------------------------------------------------------------------------
 */

#include <cmath>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"

__host__ __device__
double 
compute_join_cost(JoinRelation join_rel, 
                    RelationID left_id, JoinRelation left_rel,
                    RelationID right_id, JoinRelation right_rel,
                    BaseRelation* base_rels, int number_of_rels
                )
{
    // this cost function represents the "cost" of an hash join
    // once again, this is pretty random
    return join_rel.rows + left_rel.cost + right_rel.cost;
}

__host__ __device__
unsigned int 
estimate_join_rows(JoinRelation join_rel, 
                    RelationID &left_id, JoinRelation &left_rel,
                    RelationID &right_id, JoinRelation &right_rel,
                    BaseRelation* base_rels, int number_of_rels
                ) 
{
    double sel = 1.0;
    
    // for each edge of the left relation that gets into the right relation
    // I divide selectivity by the number of baserel tuples
    // This is quick and dirty but also wrong: in theory I sould check which
    // of the two relation the index refers to and use the number of tuples
    // of that table.
    for (int i = 1; i <= number_of_rels; i++){
        int base_relid = 1<<i;
        BaseRelation baserel = base_rels[i-1];
        if (left_rel.edges & right_id & base_relid){
            sel *= 1.0 / baserel.tuples;
        }
    }
    
    double rows = sel * (double) left_rel.rows * (double) right_rel.rows;
    return rows > 1 ? round(rows) : 1;
}


__device__
JoinRelation joinCost::operator()(JoinRelation jr){
    RelationID left_id = memo_keys[jr.left_relation_idx];
    RelationID right_id = memo_keys[jr.right_relation_idx];
    JoinRelation left_rel = memo_vals[jr.left_relation_idx];
    JoinRelation right_rel = memo_vals[jr.right_relation_idx];

    jr.rows = estimate_join_rows(jr, left_id, left_rel, right_id, right_rel,
                                base_rels.get(), n_rels);

    jr.cost = compute_join_cost(jr, left_id, left_rel, right_id, right_rel,
                                base_rels.get(), n_rels);

    return jr;
}
