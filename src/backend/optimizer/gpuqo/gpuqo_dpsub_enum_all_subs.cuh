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

template<typename BitmapsetN, bool NO_CCC>
struct dpsubEnumerateAllSubs{
    __device__
    JoinRelation<BitmapsetN> operator()(BitmapsetN relid, 
                                    uint32_t cid, int n_splits,
                                    HashTableDpsub<BitmapsetN> &memo, GpuqoPlannerInfo<BitmapsetN>* info)
    {
        JoinRelation<BitmapsetN> jr_out;
        jr_out.cost.total = INFF;
        jr_out.cost.startup = INFF;
        int qss = relid.size();
        uint_t<BitmapsetN> n_possible_joins = ((uint_t<BitmapsetN>)1)<<qss;
        uint_t<BitmapsetN> n_pairs = ceil_div(n_possible_joins, n_splits);
        uint_t<BitmapsetN> join_id = (cid)*n_pairs;
        BitmapsetN l = expandToMask(BitmapsetN(join_id), relid);
        BitmapsetN r;

        LOG_DEBUG("[%u, %u] n_splits=%d\n", relid.toUint(), cid, n_splits);

        Assert(blockDim.x == BLOCK_DIM);
        volatile __shared__ BitmapsetN ctxStack[BLOCK_DIM];
        ccc_stack_t<BitmapsetN> stack;
        stack.ctxStack = ctxStack;
        stack.stackTop = 0;

        bool valid = join_id < n_possible_joins;
        for (int i = 0; i < n_pairs; i++){
            r = relid - l;
            
            try_join<BitmapsetN,true,true,NO_CCC>(relid, jr_out, l, r, valid, 
                                                    stack, memo, info);

            l = nextSubset(l, relid);

            // if l becomes 0, I reached the end and I mark all next joins as
            // invalid
            valid = valid && (l != 0);
        }

        if (LANE_ID < stack.stackTop){
            int pos = W_OFFSET + stack.stackTop - LANE_ID - 1;
            BitmapsetN l = stack.ctxStack[pos];
            BitmapsetN r = relid - l;

            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, l.toUint(), r.toUint());

            JoinRelation<BitmapsetN> left_rel = *memo.lookup(l);
            JoinRelation<BitmapsetN> right_rel = *memo.lookup(r);
            do_join(jr_out, l, left_rel, r, right_rel, info);
        }

        return jr_out;
    }
};

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH