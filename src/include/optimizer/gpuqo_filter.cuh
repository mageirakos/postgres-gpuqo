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
                    BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table);
extern __host__ __device__
RelationID get_neighbours(RelationID set, BaseRelation* base_rels, int n_rels);

extern __host__ __device__
bool is_connected(RelationID relid,
                    BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table);

struct filterJoinedDisconnected : public thrust::unary_function<thrust::tuple<RelationID, JoinRelation>, bool>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    thrust::device_ptr<EdgeInfo> edge_table;
    int n_rels;
public:
    filterJoinedDisconnected(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        int _n_rels,
        thrust::device_ptr<EdgeInfo> _edge_table
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), base_rels(_base_rels),
        edge_table(_edge_table), n_rels(_n_rels)
    {}

    __device__
    bool operator()(thrust::tuple<RelationID, JoinRelation> t);
};

struct filterDisconnectedRelations : public thrust::unary_function<RelationID, bool>
{
    thrust::device_ptr<BaseRelation> base_rels;
    thrust::device_ptr<EdgeInfo> edge_table;
    int n_rels;
public:
    filterDisconnectedRelations(
        thrust::device_ptr<BaseRelation> _base_rels,
        int _n_rels,
        thrust::device_ptr<EdgeInfo> _edge_table
    ) : base_rels(_base_rels), edge_table(_edge_table), n_rels(_n_rels)
    {}

    __device__
    bool operator()(RelationID relid);
};
	
#endif							/* GPUQO_FILTER_CUH */
