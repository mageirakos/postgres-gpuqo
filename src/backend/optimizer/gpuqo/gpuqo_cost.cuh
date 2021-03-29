/*------------------------------------------------------------------------
 *
 * gpuqo_cost.cuh
 *      definition of the common cost-computing function
 *
 * src/backend/optimizer/gpuqo/gpuqo_cost.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_COST_CUH
#define GPUQO_COST_CUH

#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"

#define BASEREL_COEFF   0.2
#define HASHJOIN_COEFF  1
#define INDEXSCAN_COEFF 2
#define SORT_COEFF      2

__host__ __device__
static bool has_useful_index(RelationID left_rel_id, RelationID right_rel_id, 
                            GpuqoPlannerInfo* info){
    if (BMS32_SIZE(right_rel_id) != 1)  // inner must be base rel
        return false;
    // -1 since it's 1-indexed, 
    // another -1 since relation with id 0b10 is at index 0 and so on
    int baserel_right_idx = BMS32_LOWEST_POS(right_rel_id) - 2;
    
    return BMS32_INTERSECTS(
        left_rel_id, 
        info->indexed_edge_table[baserel_right_idx]
    );
}

__host__ __device__
static float baserel_cost(BaseRelation &base_rel){
    return BASEREL_COEFF * base_rel.tuples;
}

__host__ __device__
static float 
calc_join_cost(RelationID left_rel_id, JoinRelation &left_rel,
                RelationID right_rel_id, JoinRelation &right_rel,
                float join_rel_rows, GpuqoPlannerInfo* info)
{
    float hj_cost, nl_cost, inl_cost, sm_cost;
    float min_cost;

    // hash join
    hj_cost = HASHJOIN_COEFF * join_rel_rows + left_rel.cost + right_rel.cost;
    min_cost = hj_cost;

    // nested loop join
    nl_cost = left_rel.cost + left_rel.rows * right_rel.cost;
    min_cost = min(min_cost, nl_cost);
    
    // indexed nested loop join
    if (has_useful_index(left_rel_id, right_rel_id, info)){
        inl_cost = left_rel.cost 
            + INDEXSCAN_COEFF * left_rel.rows * max(
                                            join_rel_rows/left_rel.rows, 
                                            1.0);
        min_cost = min(min_cost, inl_cost);
    }

    // explicit sort merge
    sm_cost = left_rel.cost + right_rel.cost
        + SORT_COEFF * left_rel.rows * log(left_rel.rows)
        + SORT_COEFF * right_rel.rows * log(right_rel.rows);
    min_cost = min(min_cost, sm_cost);

    return min_cost;
}

__host__ __device__
static float 
estimate_join_selectivity(RelationID left_rel_id, JoinRelation &left_rel,
                        RelationID right_rel_id, JoinRelation &right_rel,
                        GpuqoPlannerInfo* info)
{
    float sel = 1.0;

    // check fk with base relations
    if (BMS32_SIZE(left_rel_id) == 1 && BMS32_SIZE(right_rel_id) == 1){
        int left_rel_idx = BMS32_LOWEST_POS(left_rel_id)-2;
        int right_rel_idx = BMS32_LOWEST_POS(right_rel_id)-2;
        float fksel = info->fk_selecs[left_rel_idx * info->n_rels + right_rel_idx];

        if(!isnan(fksel)){
            return fksel;
        }
    }
    
    // for each ec that involves any baserel on the left and on the right,
    // get its selectivity.
    // NB: one equivalence class may only apply a selectivity once so the lowest
    // matching id on both sides is kept
    EqClassInfo* ec = info->eq_classes;
    while (ec != NULL){
        RelationID match_l = BMS32_INTERSECTION(ec->relids, left_rel_id);
        RelationID match_r = BMS32_INTERSECTION(ec->relids, right_rel_id);

        if (match_l != BMS32_EMPTY && match_r != BMS32_EMPTY){
            // more than one on the same equivalence class may match
            // just take the lowest one (already done in BMS32_SET_ALL_LOWER)

            int idx_l = BMS32_SIZE(
                BMS32_INTERSECTION(
                    BMS32_SET_ALL_LOWER(match_l),
                    ec->relids
                )
            );
            int idx_r = BMS32_SIZE(
                BMS32_INTERSECTION(
                    BMS32_SET_ALL_LOWER(match_r),
                    ec->relids
                )
            );
            int size = BMS32_SIZE(ec->relids);

            sel *= ec->sels[idx_l*size+idx_r];
        }
        ec = ec->next;
    }
    
    return sel;
}

__host__ __device__
static float 
estimate_join_rows(RelationID left_rel_id, JoinRelation &left_rel,
                RelationID right_rel_id, JoinRelation &right_rel,
                GpuqoPlannerInfo* info)
{
    float sel = estimate_join_selectivity(left_rel_id, left_rel, 
                                            right_rel_id, right_rel, info);
    float rows = sel * left_rel.rows * right_rel.rows;

    // clamp the number of rows
    return rows > 1 ? round(rows) : 1;
}

#endif
