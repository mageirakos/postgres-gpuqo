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
    JoinRelation operator()(RelationID relid, uint64_t cid)
    {
        JoinRelation jr_out;
        jr_out.id = BMS64_EMPTY;
        jr_out.cost = INFD;
        int qss = BMS64_SIZE(relid);
        uint64_t n_possible_joins = 1ULL<<qss;
        uint64_t n_pairs = ceil_div(n_possible_joins, n_splits);
        uint64_t join_id = (cid)*n_pairs;
        RelationID l = BMS64_EXPAND_TO_MASK(join_id, relid);
        RelationID r;

        LOG_DEBUG("[%llu, %llu] n_splits=%d\n", relid, cid, n_splits);

        Assert(blockDim.x == BLOCK_DIM);
        volatile __shared__ join_stack_elem_t ctxStack[BLOCK_DIM];
        join_stack_t stack;
        stack.ctxStack = ctxStack;
        stack.stackTop = 0;
        stack.wOffset = threadIdx.x & (~(WARP_SIZE-1));
        stack.lane_id = threadIdx.x & (WARP_SIZE-1);
        stack.lanemask_le = (1 << (stack.lane_id+1)) - 1;

        bool stop = false;
        for (int i = 0; i < n_pairs; i++){
            stop = stop || (join_id+i != 0 && l == 0) || (join_id+i > n_possible_joins);

            if (stop){
                r=0; 
                // makes try_join process an invalid pair, giving it the possibility
                // to pop an element from the stack 
            } else {
                r = BMS64_DIFFERENCE(relid, l);
            }
            
            try_join(relid, jr_out, l, r, true, stack, memo_vals.get(), info);

            l = BMS64_NEXT_SUBSET(l, relid);
        }

        if (stack.lane_id < stack.stackTop){
            int pos = stack.wOffset + stack.stackTop - stack.lane_id - 1;
            JoinRelation *left_rel = stack.ctxStack[pos].left_rel;
            JoinRelation *right_rel = stack.ctxStack[pos].right_rel;

            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%llu, r=%llu\n", stack.wOffset, stack.lane_id, pos, left_rel->id, right_rel->id);

            do_join(relid, jr_out, *left_rel, *right_rel, info);
        }

        return jr_out;
    }
};

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH