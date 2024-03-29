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


template<typename BitmapsetN>
__host__ __device__
static struct PathCost
cost_baserel(BaseRelation<BitmapsetN> &base_rel){
    return (struct PathCost) {
        .startup = 0.0f,
        .total = base_rel.composite ? base_rel.cost.total : 0.0f
    };
}


template <typename BitmapsetN>
__host__ __device__
static struct PathCost
cost_nestloop(BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                CostExtra extra, GpuqoPlannerInfo<BitmapsetN>* info)
{
    return (struct PathCost) {
        .startup = 0.0f,
        .total = INFF
    };
}

template <typename BitmapsetN>
__host__ __device__
static struct PathCost
cost_hashjoin(BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                CostExtra extra, GpuqoPlannerInfo<BitmapsetN>* info)
{
    return (struct PathCost) {
        .startup = 0.0f,
        .total = extra.joinrows + inner_rel.cost.total + outer_rel.cost.total
    };
}

#endif
