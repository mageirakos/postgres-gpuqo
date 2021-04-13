/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_enum_all_subs.cuh
 *      "All-subs" enumeration functor for DPsub
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_enum_all_subs.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_ENUM_ALL_SUBS_CUH
#define GPUQO_DPSUB_ENUM_ALL_SUBS_CUH

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_dpsub.cuh"

__device__
static JoinRelation dpsubEnumerateAllSubs(RelationID relid, uint32_t cid, 
                                int n_splits,
                                HashTableType &memo, GpuqoPlannerInfo* info)
{
    JoinRelation jr_out;
    jr_out.cost = INFD;
    int qss = relid.size();
    uint32_t n_possible_joins = 1U<<qss;
    uint32_t n_pairs = ceil_div(n_possible_joins, n_splits);
    uint32_t join_id = (cid)*n_pairs;
    RelationID l = expandToMask(RelationID(join_id), relid);
    RelationID r;

    LOG_DEBUG("[%u, %u] n_splits=%d\n", relid.toUint(), cid, n_splits);

    Assert(blockDim.x == BLOCK_DIM);
    volatile __shared__ join_stack_elem_t ctxStack[BLOCK_DIM];
    join_stack_t stack;
    stack.ctxStack = ctxStack;
    stack.stackTop = 0;

    bool valid = join_id < n_possible_joins;
    for (int i = 0; i < n_pairs; i++){
        r = relid - l;
        
        try_join<true,true>(relid, jr_out, l, r, valid, stack, memo, info);

        l = nextSubset(l, relid);

        // if l becomes 0, I reached the end and I mark all next joins as
        // invalid
        valid = valid && (l != 0);
    }

    if (LANE_ID < stack.stackTop){
        int pos = W_OFFSET + stack.stackTop - LANE_ID - 1;
        RelationID l = stack.ctxStack[pos];
        RelationID r = relid - l;

        LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, l.toUint(), r.toUint());

        JoinRelation left_rel = *memo.lookup(l);
        JoinRelation right_rel = *memo.lookup(r);
        do_join(jr_out, l, left_rel, r, right_rel, info);
    }

    return jr_out;
}

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH