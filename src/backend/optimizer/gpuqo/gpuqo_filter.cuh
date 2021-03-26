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


/**
 * Grow `from` following all its neighbours within `subset`.
 * 
 * NB: `from` must be included in `subset`.
 */
__host__ __device__
__forceinline__
RelationID grow(RelationID from, RelationID subset, EdgeMask* edge_table)
{
    RelationID V = BMS32_EMPTY;
    RelationID N = from;

    Assert(BMS32_IS_SUBSET(from, subset));

    // as long as there are nodes to visit
    while (N != BMS32_EMPTY) {
        // pop one of the not visited neighbours 
        int baserel_idx = BMS32_LOWEST_POS(N)-2;

        LOG_DEBUG("[%u] N=%u V=%u i=%d\n", relid, N, V, baserel_idx);

        // mark as visited
        V = BMS32_SET(V, baserel_idx+1);
        
        // add his neighbours to the nodes to visit
        N = BMS32_UNION(N, edge_table[baserel_idx]);

        // keep only permitted nodes 
        N = BMS32_INTERSECTION(N, subset);

        // remove already visited nodes (including baserel_idx)
        N = BMS32_DIFFERENCE(N, V);
    };

    return V;
}

__host__ __device__
__forceinline__
bool is_connected(RelationID relid, EdgeMask* edge_table)
{
    if (relid == BMS32_EMPTY){
        return false;
    }

    return grow(BMS32_LOWEST(relid), relid, edge_table) == relid;
}

/**
 * Returns true if the subgraph contining the nodes in relid is cyclic.
 * 
 * NB: This algorithm relies on the BFS-order of base relation indices.
 * TODO: improve by cycling only on present base relations.
 */
__host__ __device__
__forceinline__
bool is_cyclic(RelationID relid, GpuqoPlannerInfo *info)
{
    // for each base relation
    for (int i=0; i<info->n_rels; i++){
        RelationID r = BMS32_NTH(i+1);

        // if it is in relid
        if (BMS32_INTERSECTS(relid, r)){
            // check that there is at most one backwards arc
            if (BMS32_SIZE(BMS32_INTERSECTION(
                    BMS32_INTERSECTION(relid, BMS32_SET_ALL_LOWER_INC(r)),
                    info->edge_table[i]
                )) > 1
            ){
                // found a cycle
                return true;
            }
        }
    }

    // no cycles
    return false;
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

struct findCycleInRelation : public thrust::unary_function<RelationID, bool>
{
    GpuqoPlannerInfo* info;
public:
    findCycleInRelation(
        GpuqoPlannerInfo* _info
    ) : info(_info)
    {}

    __device__
    bool operator()(RelationID relid) 
    {
        return is_cyclic(relid, info);
    }
};
	
#endif							/* GPUQO_FILTER_CUH */
