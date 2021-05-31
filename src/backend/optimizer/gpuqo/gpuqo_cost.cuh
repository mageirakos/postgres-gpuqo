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
#include "gpuqo_row_estimation.cuh"

#define BASEREL_COEFF   0.2
#define HASHJOIN_COEFF  1
#define INDEXSCAN_COEFF 2
#define SORT_COEFF      2

#define COST_FUNCTION_OVERHEAD 3000L

template<typename BitmapsetN>
__host__ __device__
static bool has_useful_index(BitmapsetN left_rel_id, BitmapsetN right_rel_id, 
                            GpuqoPlannerInfo<BitmapsetN>* info){
    if (right_rel_id.size() != 1)  // inner must be base rel
        return false;

    // -1 since it's 1-indexed, 
    // another -1 since relation with id 0b10 is at index 0 and so on
    int baserel_right_idx = right_rel_id.lowestPos() - 1;

    if (info->base_rels[baserel_right_idx].composite)
        return false;
    
    return left_rel_id.intersects(info->indexed_edge_table[baserel_right_idx]);
}

template<typename BitmapsetN>
__host__ __device__
static float baserel_cost(BaseRelation<BitmapsetN> &base_rel){
    return BASEREL_COEFF * base_rel.tuples;
}

template<typename BitmapsetN>
__host__ __device__
static float 
calc_join_cost(BitmapsetN left_rel_id, JoinRelation<BitmapsetN> &left_rel,
                BitmapsetN right_rel_id, JoinRelation<BitmapsetN> &right_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
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
        + SORT_COEFF * left_rel.rows * logf(left_rel.rows)
        + SORT_COEFF * right_rel.rows * logf(right_rel.rows);
    min_cost = min(min_cost, sm_cost);

#if defined(SIMULATE_COMPLEX_COST_FUNCTION) && COST_FUNCTION_OVERHEAD>0
    //Additional overhead to simulate complex cost functions
        volatile long counter = COST_FUNCTION_OVERHEAD;
        while (counter > 0){
            --counter;
        }
#endif

    return min_cost;
}

#endif
