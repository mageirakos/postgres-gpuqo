/*-------------------------------------------------------------------------
 *
 * gpuqo_cost.cuh
 *	  declaration of gpuqo cost related functions.
 *
 * src/include/optimizer/gpuqo_debug.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_COST_CUH
#define GPUQO_COST_CUH

#include "gpuqo.cuh"

#define BASEREL_COEFF   0.2
#define HASHJOIN_COEFF  1
#define INDEXSCAN_COEFF 2
#define SORT_COEFF      2

extern __host__ __device__
float baserel_cost(BaseRelation &base_rel);

extern __host__ __device__
float compute_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, GpuqoPlannerInfo* info
);

extern __host__ __device__
float estimate_join_rows(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, GpuqoPlannerInfo* info
);
	
#endif							/* GPUQO_COST_CUH */
