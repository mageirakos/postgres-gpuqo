/*------------------------------------------------------------------------
 *
 * gpuqo_filter.cu
 *      definition of the common filtering functions
 *
 * src/backend/optimizer/gpuqo/gpuqo_filter.cu
 *
 *-------------------------------------------------------------------------
 */

 #include <cmath>
 #include <cstdint>
 
 #include "optimizer/gpuqo_common.h"
 
 #include "optimizer/gpuqo.cuh"
 #include "optimizer/gpuqo_timing.cuh"
 #include "optimizer/gpuqo_debug.cuh"
 #include "optimizer/gpuqo_filter.cuh"

__host__ __device__
bool is_disjoint(JoinRelation &left_rel, JoinRelation &right_rel)
{
    return !BMS64_INTERSECTS(left_rel.id, right_rel.id);
}

__host__ __device__
bool are_connected(JoinRelation &left_rel, JoinRelation &right_rel,
                    BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table)
{
    return (left_rel.edges & right_rel.id) != 0ULL;
}

__host__ __device__
RelationID get_neighbours(RelationID set, BaseRelation* base_rels, int n_rels){
    RelationID neigs = BMS64_EMPTY;
    while (set != BMS64_EMPTY){
        int baserel_idx = BMS64_LOWEST_POS(set)-2;
        neigs = BMS64_UNION(neigs, base_rels[baserel_idx].edges);
        set = BMS64_UNSET(set, baserel_idx+1);
    }
    return BMS64_DIFFERENCE(neigs, set);
}

__host__ __device__
bool is_connected(RelationID relid,
                    BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table)
{
    RelationID T = BMS64_LOWEST(relid);
    RelationID N = T;
    do {
        // explore only from newly found nodes that are missing
        N = BMS64_INTERSECTION(
            BMS64_DIFFERENCE(relid, T), 
            get_neighbours(N, base_rels, n_rels)
        );
        // add new nodes to set
        T = BMS64_UNION(T, N);
    } while (T != relid && N != BMS64_EMPTY);
    // either all nodes have been found or no new connected node exists

    // if I managed to visit all nodes, then subgraph is connected
    return T == relid; 
}
 
__device__
bool filterJoinedDisconnected::operator()(thrust::tuple<RelationID, JoinRelation> t) 
{
    RelationID relid = t.get<0>();
    JoinRelation jr = t.get<1>();

    LOG_DEBUG("%llu %llu\n", 
        jr.left_relation_idx,
        jr.right_relation_idx
    );

    JoinRelation left_rel = memo_vals[jr.left_relation_idx];
    JoinRelation right_rel = memo_vals[jr.right_relation_idx];

    if (!is_disjoint(left_rel, right_rel)) // not disjoint
        return true;
    else{
        return !are_connected(left_rel, right_rel, base_rels.get(), n_rels, edge_table.get());
    }
}

__device__
bool filterDisconnectedRelations::operator()(RelationID relid) 
{
    return !is_connected(relid, 
            base_rels.get(), n_rels, edge_table.get());
}
