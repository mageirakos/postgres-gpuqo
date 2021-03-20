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


struct dpsubEnumerateAllSubs : public pairs_enum_func_t 
{
    HashTable32bit memo;
    GpuqoPlannerInfo* info;
    int n_splits;
public:
    dpsubEnumerateAllSubs(
        HashTable32bit _memo,
        GpuqoPlannerInfo* _info,
        int _n_splits
    ) : memo(_memo), info(_info), n_splits(_n_splits)
    {}

    __device__
    JoinRelation operator()(RelationID relid, uint32_t cid)
    {
        JoinRelation jr_out;
        jr_out.id = BMS32_EMPTY;
        jr_out.cost = INFD;
        int qss = BMS32_SIZE(relid);
        uint32_t n_possible_joins = 1U<<qss;
        uint32_t n_pairs = ceil_div(n_possible_joins, n_splits);
        uint32_t join_id = (cid)*n_pairs;
        RelationID l = BMS32_EXPAND_TO_MASK(join_id, relid);
        RelationID r;

        LOG_DEBUG("[%u, %u] n_splits=%d\n", relid, cid, n_splits);

        Assert(blockDim.x == BLOCK_DIM);
        volatile __shared__ join_stack_elem_t ctxStack[BLOCK_DIM];
        join_stack_t stack;
        stack.ctxStack = ctxStack;
        stack.stackTop = 0;

        bool valid = join_id < n_possible_joins;
        for (int i = 0; i < n_pairs; i++){
            r = BMS32_DIFFERENCE(relid, l);
            
            try_join<true>(jr_out, l, r, valid, stack, memo, info);

            l = BMS32_NEXT_SUBSET(l, relid);

            // if l becomes 0, I reached the end and I mark all next joins as
            // invalid
            valid = valid && (l != 0);
        }

        if (LANE_ID < stack.stackTop){
            int pos = W_OFFSET + stack.stackTop - LANE_ID - 1;
            RelationID l = stack.ctxStack[pos];
            RelationID r = BMS32_DIFFERENCE(relid, l);

            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, l, r);

            JoinRelation *left_rel = memo.lookup(l);
            JoinRelation *right_rel = memo.lookup(r);
            do_join(jr_out, *left_rel, *right_rel, info);
        }

        return jr_out;
    }
};

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH