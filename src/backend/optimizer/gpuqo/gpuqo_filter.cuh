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

template<typename BitmapsetN>
__host__ __device__
__forceinline__
bool is_disjoint(BitmapsetN left_rel_id, BitmapsetN right_rel_id)
{
    return !left_rel_id.intersects(right_rel_id);
}

template<typename BitmapsetN>
__host__ __device__
__forceinline__
bool is_disjoint_rel(JoinRelationDetailed<BitmapsetN> &left_rel, 
                 JoinRelationDetailed<BitmapsetN> &right_rel)
{ 
    return is_disjoint(left_rel.id, right_rel.id);
}

template<typename BitmapsetN>
__host__ __device__
__forceinline__
bool is_disjoint(JoinRelation<BitmapsetN> &join_rel)
{
    return is_disjoint(
        join_rel.left_rel_id, 
        join_rel.right_rel_id
    );
}

template<typename BitmapsetN>
__host__ __device__
__forceinline__
bool are_connected(BitmapsetN left_edges, BitmapsetN right_id,
                GpuqoPlannerInfo<BitmapsetN>* info)
{
    return left_edges.intersects(right_id);
}

template<typename BitmapsetN>
__host__ __device__
__forceinline__
bool are_connected_rel(JoinRelationDetailed<BitmapsetN> &left_rel, 
                    JoinRelationDetailed<BitmapsetN> &right_rel,
                    GpuqoPlannerInfo<BitmapsetN>* info)
{
    return are_connected(left_rel.edges, right_rel.id, info);
}

template<typename BitmapsetN>
__host__ __device__
__forceinline__
BitmapsetN get_neighbours(BitmapsetN set, BitmapsetN* edge_table){
    BitmapsetN neigs = BitmapsetN(0);
    BitmapsetN temp = set;
    while (!temp.empty()){
        int baserel_idx = temp.lowestPos()-1;
        neigs |= edge_table[baserel_idx];
        temp.unset(baserel_idx+1);
    }
    return neigs - set;
}


/**
 * Grow `from` following all its neighbours within `subset`.
 * 
 * NB: `from` must be included in `subset`.
 */
 template<typename BitmapsetN>
__host__ __device__
__forceinline__
BitmapsetN grow(BitmapsetN from, BitmapsetN subset, BitmapsetN* edge_table)
{
    BitmapsetN V = BitmapsetN(0);
    BitmapsetN N = from;

    LOG_DEBUG("grow(%u, %u)\n", from.toUint(), subset.toUint());

    Assert(from.isSubset(subset));

    // as long as there are nodes to visit
    while (!N.empty()) {
        // pop one of the not visited neighbours 
        int baserel_idx = N.lowestPos()-1;

        LOG_DEBUG("[%u] N=%u V=%u i=%d\n", from.toUint(), N.toUint(), V.toUint(), baserel_idx);

        // mark as visited
        V.set(baserel_idx+1);
        
        // add his neighbours to the nodes to visit
        N |= edge_table[baserel_idx];

        // keep only permitted nodes 
        N &= subset;

        // remove already visited nodes (including baserel_idx)
        N -= V;
    };

    return V;
}

template<typename BitmapsetN>
__host__ __device__
__forceinline__
bool is_connected(BitmapsetN relid, BitmapsetN* edge_table)
{
    if (relid.empty()){
        return false;
    }

    return grow(relid.lowest(), relid, edge_table) == relid;
}

/**
 * Returns true if the subgraph contining the nodes in relid is cyclic.
 * 
 * NB: This algorithm relies on the BFS-order of base relation indices.
 * TODO: improve by cycling only on present base relations.
 */
template<typename BitmapsetN>
__host__ __device__
__forceinline__
bool is_cyclic(BitmapsetN relid, GpuqoPlannerInfo<BitmapsetN> *info)
{
    // for each base relation
    for (int i=0; i<info->n_rels; i++){
        BitmapsetN r = BitmapsetN::nth(i+1);

        // if it is in relid
        if (relid.intersects(r)){
            // check that there is at most one backwards arc
            if ((relid & r.allLowerInc() & info->edge_table[i]).size() > 1){
                // found a cycle
                return true;
            }
        }
    }

    // no cycles
    return false;
}

template<typename BitmapsetN>
struct filterDisconnectedRelations : public thrust::unary_function<BitmapsetN, bool>
{
    GpuqoPlannerInfo<BitmapsetN>* info;
public:
    filterDisconnectedRelations(
        GpuqoPlannerInfo<BitmapsetN>* _info
    ) : info(_info)
    {}

    __device__
    bool operator()(BitmapsetN relid) 
    {
        return !is_connected(relid, info->edge_table);
    }
};

template<typename BitmapsetN>
struct findCycleInRelation : public thrust::unary_function<BitmapsetN, bool>
{
    GpuqoPlannerInfo<BitmapsetN>* info;
public:
    findCycleInRelation(
        GpuqoPlannerInfo<BitmapsetN>* _info
    ) : info(_info)
    {}

    __device__
    bool operator()(BitmapsetN relid) 
    {
        return is_cyclic(relid, info);
    }
};
	
#endif							/* GPUQO_FILTER_CUH */
