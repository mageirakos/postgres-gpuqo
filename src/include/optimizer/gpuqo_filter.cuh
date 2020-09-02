/*-------------------------------------------------------------------------
 *
 * gpuqo_filter.cuh
 *	  declaration of gpuqo filtering-related functions.
 *
 * src/include/optimizer/gpuqo_filter.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_FILTER_CUH
#define GPUQO_FILTER_CUH

#include <thrust/device_ptr.h>

#include "optimizer/gpuqo_common.h"
#include "optimizer/gpuqo.cuh"

extern __host__ __device__
bool is_disjoint(JoinRelation &left_rel, JoinRelation &right_rel);

extern __host__ __device__
bool are_connected(JoinRelation &left_rel, JoinRelation &right_rel,
                    GpuqoPlannerInfo* info);
extern __host__ __device__
RelationID get_neighbours(RelationID set, GpuqoPlannerInfo* info);

extern __host__ __device__
bool is_connected(RelationID relid,
                    GpuqoPlannerInfo* info);

struct filterJoinedDisconnected : public thrust::unary_function<thrust::tuple<RelationID, JoinRelation>, bool>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    GpuqoPlannerInfo* info;
public:
    filterJoinedDisconnected(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        GpuqoPlannerInfo* _info
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), info(_info)
    {}

    __device__
    bool operator()(thrust::tuple<RelationID, JoinRelation> t);
};

struct filterDisconnectedRelations : public thrust::unary_function<RelationID, bool>
{
    GpuqoPlannerInfo* info;
public:
    filterDisconnectedRelations(
        GpuqoPlannerInfo* _info
    ) : info(_info)
    {}

    __device__
    bool operator()(RelationID relid);
};
	
#endif							/* GPUQO_FILTER_CUH */
