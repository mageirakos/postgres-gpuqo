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
bool is_disjoint(RelationID left_rel_id, RelationID right_rel_id)
{
    return !BMS32_INTERSECTS(left_rel_id, right_rel_id);
}

__host__ __device__
__forceinline__
bool is_disjoint(JoinRelation &left_rel, JoinRelation &right_rel)
{ 
    return is_disjoint(left_rel.id, right_rel.id);
}

__host__ __device__
__forceinline__
bool is_disjoint(JoinRelation &join_rel)
{
    return is_disjoint(
        join_rel.left_relation_id, 
        join_rel.right_relation_id
    );
}

__host__ __device__
__forceinline__
bool are_connected(EdgeMask left_edges, RelationID right_id,
                GpuqoPlannerInfo* info)
{
    return BMS32_INTERSECTS(left_edges, right_id);
}

__host__ __device__
__forceinline__
bool are_connected(JoinRelation &left_rel, JoinRelation &right_rel,
                GpuqoPlannerInfo* info)
{
    return are_connected(left_rel.edges, right_rel.id, info);
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
