/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_tree.cuh
 *      enumeration functors for DPsub when subgraph is a tree
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_tree.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_ENUM_TREE_CUH
#define GPUQO_DPSUB_ENUM_TREE_CUH

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_dpsub.cuh"

__device__
static JoinRelation dpsubEnumerateTreeSimple(RelationID relid, 
                        uint32_t cid, int n_splits,
                        HashTableType &memo, GpuqoPlannerInfo* info)
{
    JoinRelation jr_out;
    jr_out.cost = INFD;

    uint32_t n_possible_joins = relid.size();

    for (uint32_t i = cid; i < n_possible_joins; i += n_splits){
        RelationID base_rel_id = expandToMask(RelationID::nth(i), relid);
        // TODO check if I can rely on this in general...
        RelationID permitted = relid - base_rel_id.allLower();

        LOG_DEBUG("%d %d: %u (%u)\n", 
            blockIdx.x,
            threadIdx.x,
            base_rel_id.toUint(),
            relid.toUint()
        );

        RelationID l = grow(base_rel_id, permitted, info->edge_table);
        RelationID r = relid - l;

        if (!l.empty() && !r.empty()){
            LOG_DEBUG("%d %d: %u %u (%u)\n", 
                blockIdx.x,
                threadIdx.x,
                l.toUint(),
                r.toUint(), 
                relid.toUint()
            );

            JoinRelation left_rel = *memo.lookup(l);
            JoinRelation right_rel = *memo.lookup(r);
            do_join(jr_out, l, left_rel, r, right_rel, info);
            do_join(jr_out, r, right_rel, l, left_rel, info);
        }
    }

    return jr_out;
}

__device__
static JoinRelation dpsubEnumerateTreeWithSubtrees(RelationID relid, 
                        uint32_t cid, int n_splits,
                        HashTableType &memo, GpuqoPlannerInfo* info)
{ 
    JoinRelation jr_out;
    jr_out.cost = INFD;

    uint32_t n_possible_joins = relid.size();

    for (uint32_t i = cid; i < n_possible_joins; i += n_splits){
        RelationID base_rel_id = expandToMask(RelationID::nth(i), relid);
        int base_rel_idx = base_rel_id.lowestPos()-1;

        Assert(base_rel_idx < info->n_rels);

        RelationID S = info->subtrees[base_rel_idx] & relid;

        RelationID l = S;
        RelationID r = relid - S;

        LOG_DEBUG("%d %d: %u %u (%u)\n", 
            blockIdx.x,
            threadIdx.x,
            l.toUint(),
            r.toUint(), 
            relid.toUint()
        );

        if (!l.empty() && !r.empty()){
            JoinRelation left_rel = *memo.lookup(l);
            JoinRelation right_rel = *memo.lookup(r);

            do_join(jr_out, l, left_rel, r, right_rel, info);
            do_join(jr_out, r, right_rel, l, left_rel, info);
        }
    }

    return jr_out;
}

#endif              // GPUQO_DPSUB_ENUM_TREE_CUH