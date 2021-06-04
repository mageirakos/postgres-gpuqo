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

#endif
