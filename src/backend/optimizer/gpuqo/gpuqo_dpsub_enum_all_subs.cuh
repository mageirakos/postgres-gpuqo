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
    thrust::device_ptr<JoinRelation> memo_vals;
    GpuqoPlannerInfo* info;
    int n_splits;
public:
    dpsubEnumerateAllSubs(
        thrust::device_ptr<JoinRelation> _memo_vals,
        GpuqoPlannerInfo* _info,
        int _n_splits
    ) : memo_vals(_memo_vals), info(_info), n_splits(_n_splits)
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

        bool stop = false;
        for (int i = 0; i < n_pairs; i++){
            stop = stop || (join_id+i != 0 && l == 0) || (join_id+i > n_possible_joins);

            if (stop){
                r=0; 
                // makes try_join process an invalid pair, giving it the possibility
                // to pop an element from the stack 
            } else {
                r = BMS32_DIFFERENCE(relid, l);
            }
            
            try_join(relid, jr_out, l, r, true, stack, memo_vals.get(), info);

            l = BMS32_NEXT_SUBSET(l, relid);
        }

        if (LANE_ID < stack.stackTop){
            int pos = W_OFFSET + stack.stackTop - LANE_ID - 1;
            JoinRelation *left_rel = stack.ctxStack[pos].left_rel;
            JoinRelation *right_rel = stack.ctxStack[pos].right_rel;

            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, left_rel->id, right_rel->id);

            do_join(relid, jr_out, *left_rel, *right_rel, info);
        }

        return jr_out;
    }
};

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH