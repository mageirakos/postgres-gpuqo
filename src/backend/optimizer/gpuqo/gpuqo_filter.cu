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
 

__device__
bool filterJoinedDisconnected::operator()(thrust::tuple<RelationID, JoinRelation> t) 
{
    RelationID relid = t.get<0>();
    JoinRelation jr = t.get<1>();

#ifdef GPUQO_DEBUG
    printf("%llu %llu\n", 
        jr.left_relation_idx,
        jr.right_relation_idx
    );
#endif

    JoinRelation left_rel = memo_vals[jr.left_relation_idx];
    JoinRelation right_rel = memo_vals[jr.right_relation_idx];

    if (!is_disjoint(left_rel, right_rel)) // not disjoint
        return true;
    else{
        return !are_connected(left_rel, right_rel, base_rels.get(), n_rels, edge_table.get());
    }
}