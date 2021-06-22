/*------------------------------------------------------------------------
 *
 * gpuqo_cost_simple.cuh
 *      definition of a simplified cost function
 *
 * src/backend/optimizer/gpuqo/gpuqo_cost_simple.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_COST_SIMPLE_CUH
#define GPUQO_COST_SIMPLE_CUH

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
static struct PathCost
cost_baserel(BaseRelation<BitmapsetN> &base_rel){
    return (struct PathCost) {
        .startup = 0.0f,
        .total = base_rel.composite ? base_rel.cost.total : BASEREL_COEFF * base_rel.tuples;
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
        .total = outer_rel.cost.total + outer_rel.rows * inner_rel.cost.total
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
        .total = HASHJOIN_COEFF * extra.joinrows + inner_rel.cost.total + outer_rel.cost.total
    };
}

#endif
