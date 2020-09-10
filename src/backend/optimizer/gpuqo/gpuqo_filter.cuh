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

#include "gpuqo.cuh"

__host__ __device__
__forceinline__
bool is_disjoint(JoinRelation &left_rel, JoinRelation &right_rel)
{
    return !BMS32_INTERSECTS(left_rel.id, right_rel.id);
}

__host__ __device__
__forceinline__
bool are_connected(JoinRelation &left_rel, JoinRelation &right_rel,
                GpuqoPlannerInfo* info)
{
    return BMS32_INTERSECTS(left_rel.edges, right_rel.id);
}

__host__ __device__
__forceinline__
RelationID get_neighbours(RelationID set, EdgeMask* edge_table){
    RelationID neigs = BMS32_EMPTY;
    RelationID temp = set;
    while (temp != BMS32_EMPTY){
        int baserel_idx = BMS32_LOWEST_POS(temp)-2;
        neigs = BMS32_UNION(neigs, edge_table[baserel_idx]);
        temp = BMS32_UNSET(temp, baserel_idx+1);
    }
    return BMS32_DIFFERENCE(neigs, set);
}

__host__ __device__
__forceinline__
bool is_connected(RelationID relid, EdgeMask* edge_table)
{
    RelationID T = BMS32_LOWEST(relid);
    RelationID N = T;
    do {
        // explore only from newly found nodes that are missing
        N = BMS32_INTERSECTION(
            BMS32_DIFFERENCE(relid, T), 
            get_neighbours(N, edge_table)
        );
        // add new nodes to set
        T = BMS32_UNION(T, N);
    } while (T != relid && N != BMS32_EMPTY);
    // either all nodes have been found or no new connected node exists

    // if I managed to visit all nodes, then subgraph is connected
    return T == relid; 
}
 
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
    bool operator()(thrust::tuple<RelationID, JoinRelation> t) 
    {
        RelationID relid = t.get<0>();
        JoinRelation jr = t.get<1>();

        LOG_DEBUG("%u %u\n", 
            jr.left_relation_idx,
            jr.right_relation_idx
        );

        JoinRelation left_rel = memo_vals[jr.left_relation_idx];
        JoinRelation right_rel = memo_vals[jr.right_relation_idx];

        if (!is_disjoint(left_rel, right_rel)) // not disjoint
            return true;
        else{
            return !are_connected(left_rel, right_rel, info);
        }
    }
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
    bool operator()(RelationID relid) 
    {
        return !is_connected(relid, info->edge_table);
    }
};
	
#endif							/* GPUQO_FILTER_CUH */
