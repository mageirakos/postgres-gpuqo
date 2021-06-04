/*------------------------------------------------------------------------
 *
 * gpuqo_cost_cout.cuh
 *      definition of the C_out cost functions
 *
 * src/backend/optimizer/gpuqo/gpuqo_cost_cout.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_COST_COUT_CUH
#define GPUQO_COST_COUT_CUH

#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_row_estimation.cuh"

#define BASEREL_COEFF   0.2f
#define HASHJOIN_COEFF  1.0f
#define INDEXSCAN_COEFF 2.0f
#define SORT_COEFF      2.0f

template<typename BitmapsetN>
__host__ __device__
static struct Cost
cost_baserel(BaseRelation<BitmapsetN> &base_rel){
    return (struct Cost) {
        .startup = 0.0f,
        .total = BASEREL_COEFF * base_rel.tuples
    };
}


template <typename BitmapsetN>
__host__ __device__
static struct Cost
cost_nestloop(BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
{
    return (struct Cost) {
        .startup = 0.0f,
    };
}

template <typename BitmapsetN>
__host__ __device__
static struct Cost
cost_hashjoin(BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
{
    return (struct Cost) {
        .startup = 0.0f,
    };
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
