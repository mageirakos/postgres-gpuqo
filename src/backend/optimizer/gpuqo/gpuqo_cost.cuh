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

#include <thrust/device_ptr.h>

#include "gpuqo.cuh"

#define BASEREL_COEFF   0.2
#define HASHJOIN_COEFF  1
#define INDEXSCAN_COEFF 2
#define SORT_COEFF      2

extern __host__ __device__
double baserel_cost(BaseRelation &base_rel);

extern __host__ __device__
double compute_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, GpuqoPlannerInfo* info
);

extern __host__ __device__
double estimate_join_rows(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, GpuqoPlannerInfo* info
);

struct joinCost : public thrust::unary_function<JoinRelation,JoinRelation>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    GpuqoPlannerInfo* info;
public:
    joinCost(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        GpuqoPlannerInfo* _info
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), info(_info)
    {}

    __device__
    JoinRelation operator()(JoinRelation jr);
};
	
#endif							/* GPUQO_COST_CUH */
