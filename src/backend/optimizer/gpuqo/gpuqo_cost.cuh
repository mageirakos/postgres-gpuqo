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

#ifdef GPUQO_COST_FUNCTION_COUT
#include "gpuqo_cost_cout.cuh"
#else
#include "gpuqo_cost_postgres.cuh"
#endif

#define COST_FUNCTION_OVERHEAD 3000L

template<typename BitmapsetN>
__host__ __device__
static struct Cost 
calc_join_cost(BitmapsetN left_rel_id, JoinRelation<BitmapsetN> &left_rel,
                BitmapsetN right_rel_id, JoinRelation<BitmapsetN> &right_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
{
    struct Cost min_cost, nlj_cost, hj_cost;

    min_cost.total = info->params.disable_cost;

    nlj_cost = cost_nestloop(left_rel_id, left_rel, right_rel_id, right_rel, join_rel_rows, info);

    if (nlj_cost.total < min_cost.total)
        min_cost = nlj_cost;

    hj_cost = cost_hashjoin(left_rel_id, left_rel, right_rel_id, right_rel, join_rel_rows, info);

    if (hj_cost.total < min_cost.total)
        min_cost = hj_cost;

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
