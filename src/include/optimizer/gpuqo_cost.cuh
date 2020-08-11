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

#include "optimizer/gpuqo_common.h"
#include "optimizer/gpuqo.cuh"

extern __host__ __device__
double compute_join_cost(JoinRelation join_rel, 
                    RelationID left_id, JoinRelation left_rel,
                    RelationID right_id, JoinRelation right_rel,
                    BaseRelation* base_rels, EdgeInfo* edge_table,
                    int number_of_rels
);

extern __host__ __device__
double estimate_join_rows(JoinRelation join_rel, 
                    RelationID &left_id, JoinRelation &left_rel,
                    RelationID &right_id, JoinRelation &right_rel,
                    BaseRelation* base_rels, EdgeInfo* edge_table,
                    int number_of_rels
);

struct joinCost : public thrust::unary_function<JoinRelation,JoinRelation>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    thrust::device_ptr<EdgeInfo> edge_table;
    int n_rels;
public:
    joinCost(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        thrust::device_ptr<EdgeInfo> _edge_table,
        int _n_rels
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), base_rels(_base_rels),
        edge_table(_edge_table), n_rels(_n_rels)
    {}

    __device__
    JoinRelation operator()(JoinRelation jr);
};
	
#endif							/* GPUQO_COST_CUH */
