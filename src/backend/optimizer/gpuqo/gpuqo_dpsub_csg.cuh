/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_csg.cuh
 *      "CSG" enumeration functor for DPsub
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_csg.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_ENUM_CSG_CUH
#define GPUQO_DPSUB_ENUM_CSG_CUH

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_dpsub.cuh"

template<typename BitmapsetN>
uint32_t dpsub_csg_evaluation(int iter, 
                            uint64_t n_remaining_sets,
                            uint64_t offset, uint32_t n_pending_sets, 
                            dpsub_iter_param_t<BitmapsetN> &params);

template<typename BitmapsetN>
struct ext_loop_stack_elem_t{
    BitmapsetN S;
    BitmapsetN X;
    BitmapsetN I;
    BitmapsetN N;
};

template<typename BitmapsetN>
struct emit_stack_elem_t{
    BitmapsetN S;
    BitmapsetN X;
    BitmapsetN I;
};

template<typename BitmapsetN>
using csg_stack_t = ccc_stack_t<emit_stack_elem_t<BitmapsetN> >;

template<typename BitmapsetN, int MAX_RELS, int STACK_SIZE>
__device__
static void enumerate_sub_csg(BitmapsetN T, BitmapsetN I, BitmapsetN E,
                    JoinRelation<BitmapsetN> &jr_out, 
                    ccc_stack_t<BitmapsetN> &join_stack,
                    HashTableDpsub<BitmapsetN> &memo, 
                    GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN, int STACK_SIZE>
__device__ 
static void enumerate_sub_csg_emit(BitmapsetN T, BitmapsetN emit_S, 
            BitmapsetN emit_X, BitmapsetN I, BitmapsetN E,
            ext_loop_stack_elem_t<BitmapsetN>* loop_stack, int &loop_stack_size,
            JoinRelation<BitmapsetN> &jr_out, 
            ccc_stack_t<BitmapsetN> &join_stack, 
            HashTableDpsub<BitmapsetN> &memo, 
            GpuqoPlannerInfo<BitmapsetN>* info);


template<typename BitmapsetN>
struct dpsubEnumerateCsg{
    __device__
    JoinRelation<BitmapsetN> operator()(BitmapsetN relid, 
                                    uint32_t cid_bits, 
                                    int n_splits, HashTableDpsub<BitmapsetN> &memo, 
                                    GpuqoPlannerInfo<BitmapsetN>* info)
    { 
        // TODO check
        Assert(n_splits % WARP_SIZE == 0 && popc(n_splits) == 1);

        BitmapsetN cid = BitmapsetN(cid_bits);

        BitmapsetN split_mask = BitmapsetN(floorPow2(n_splits)).allLower();
        BitmapsetN cmp_cid = split_mask - cid;

        JoinRelation<BitmapsetN> jr_out;
        jr_out.cost = INFD;

        volatile __shared__ BitmapsetN ctxStack[BLOCK_DIM];
        ccc_stack_t<BitmapsetN> join_stack;
        join_stack.ctxStack = ctxStack;
        join_stack.stackTop = 0;

        Assert(split_mask.size() <= relid.size());
        LOG_DEBUG("[%u, %u] split_mask=%u, cmp_cid=%u\n", 
            relid.toUint(), cid.toUint(), split_mask.toUint(), cmp_cid.toUint());

        Assert((cid|cmp_cid) == split_mask);
        Assert(!cid.intersects(cmp_cid));

        BitmapsetN inc_set = expandToMask(cid, relid);
        BitmapsetN exc_set = expandToMask(cmp_cid, relid);
        
        enumerate_sub_csg<BitmapsetN,BitmapsetN::SIZE,BitmapsetN::SIZE*2>(relid, inc_set, exc_set, jr_out, join_stack, 
            memo, info);

        if (LANE_ID < join_stack.stackTop){
            int pos = W_OFFSET + join_stack.stackTop - LANE_ID - 1;
            Assert(pos >= W_OFFSET && pos < W_OFFSET + WARP_SIZE);
            BitmapsetN l = join_stack.ctxStack[pos];
            BitmapsetN r = relid - l;

            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", 
                    W_OFFSET, LANE_ID, pos, l.toUint(), r.toUint());

            JoinRelation<BitmapsetN> left_rel = *memo.lookup(l);
            JoinRelation<BitmapsetN> right_rel = *memo.lookup(r);
            do_join(jr_out, l, left_rel, r, right_rel, info);
        }
        
        return jr_out;
    }
};

template<typename BitmapsetN, int STACK_SIZE>
__device__ 
static void enumerate_sub_csg_emit(BitmapsetN T, BitmapsetN emit_S, 
            BitmapsetN emit_X, BitmapsetN I, BitmapsetN E,
            ext_loop_stack_elem_t<BitmapsetN>* loop_stack, int &loop_stack_size,
            JoinRelation<BitmapsetN> &jr_out, ccc_stack_t<BitmapsetN> &join_stack, 
            HashTableDpsub<BitmapsetN> &memo, GpuqoPlannerInfo<BitmapsetN>* info){
    // LOG_DEBUG("enumerate_sub_csg_emit(%u, %u, %u, %u, %u)\n",
    //         T.toUint(), emit_S.toUint(), emit_X.toUint(), I.toUint(), E.toUint());
    Assert(emit_S.isSubset(T));
    Assert(I.isSubset(T));
    Assert(emit_S.empty() || is_connected(emit_S, info->edge_table));

    try_join<BitmapsetN,false,true>(T, jr_out, emit_S, T-emit_S, 
            !emit_S.empty() && I.isSubset(emit_S), 
            join_stack, memo, info);

    BitmapsetN new_N = (get_neighbours(emit_S,info->edge_table)-emit_X) & (T-E);
    
    // If possible, directly move to smaller I (it does not make 
    // sense to explore other rels in I first since it won't be 
    // possible to go back)
    BitmapsetN lowI = (I - emit_S).lowest();
    if (lowI.intersects(new_N)){
        new_N = lowI;
    }

    // do not add useless elements to stack
    if (!new_N.empty()){
        loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t<BitmapsetN>){
            emit_S, emit_X, I, new_N
        };
        Assert(loop_stack_size < STACK_SIZE);
    }
}

template<typename BitmapsetN, int MAX_RELS, int STACK_SIZE>
__device__
static void enumerate_sub_csg(BitmapsetN T, BitmapsetN I, BitmapsetN E,
                    JoinRelation<BitmapsetN> &jr_out, ccc_stack_t<BitmapsetN> &join_stack,
                    HashTableDpsub<BitmapsetN> &memo, GpuqoPlannerInfo<BitmapsetN>* info){
    BitmapsetN R = I | E;

    Assert(I.isSubset(T));

    // S, X, I, N
    ext_loop_stack_elem_t<BitmapsetN> loop_stack[STACK_SIZE];
    int loop_stack_size = 0;

    volatile __shared__ emit_stack_elem_t<BitmapsetN> ctxStack[BLOCK_DIM];
    csg_stack_t<BitmapsetN> stack;
    stack.ctxStack = ctxStack;
    stack.stackTop = 0;

    LOG_DEBUG("[%d: %d] lanemask_le=%u\n", W_OFFSET, LANE_ID, LANE_MASK_LE);

    BitmapsetN temp;
    if (!I.empty()){
        temp = I.lowest();
    } else{
        temp = T-E;
    }

    while (!temp.empty()){
        BitmapsetN v = temp.lowest();
        
        loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t<BitmapsetN>){
            BitmapsetN(0),
            v.allLowerInc(),
            I,
            v
        };

        Assert(loop_stack_size < STACK_SIZE);

        temp ^= v;
    }

    bool all_empty = false;
    while (!all_empty || stack.stackTop != 0){
        BitmapsetN emit_S;
        BitmapsetN emit_X;
        BitmapsetN next_subset = BitmapsetN(0);

        if (loop_stack_size != 0){
            ext_loop_stack_elem_t<BitmapsetN> top = loop_stack[--loop_stack_size];
            BitmapsetN S = top.S;
            BitmapsetN X = top.X;
            BitmapsetN N = top.N;
            I = top.I;
            E = R - I;

            Assert(loop_stack_size >= 0);

            // LOG_DEBUG("[%u: %u, %u] loop_stack: S=%u, X=%u, N=%u\n", T.toUint(), I.toUint(), E.toUint(), S.toUint(), X.toUint(), N.toUint());

            next_subset = nextSubset(S & N, N);

            emit_S = (S - N) | next_subset;
            emit_X = X | N;

            if (!next_subset.empty()){
                loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t<BitmapsetN>){
                    emit_S, X, I, N
                };
                Assert(loop_stack_size < STACK_SIZE);
            }
        }

        if (!all_empty){
            bool p = !next_subset.empty();
            unsigned pthBlt = __ballot_sync(WARP_MASK, !p);
            int reducedNTaken = __popc(pthBlt);
            if (LANE_ID == 0){
                LOG_DEBUG("[%d] pthBlt=%u, reducedNTaken=%d, stackTop=%d\n", W_OFFSET, pthBlt, reducedNTaken, stack.stackTop);
            }
            if (stack.stackTop >= reducedNTaken){
                int wScan = __popc(pthBlt & LANE_MASK_LE);
                int pos = W_OFFSET + stack.stackTop - wScan;
                if (!p){
                    Assert(pos >= W_OFFSET && pos < W_OFFSET + WARP_SIZE);
                    emit_S = stack.ctxStack[pos].S;
                    emit_X = stack.ctxStack[pos].X;
                    I = stack.ctxStack[pos].I;
                    E = R - I;
                    LOG_DEBUG("[%d: %d] Consuming stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S.toUint(), emit_X.toUint(), I.toUint());
                }
                stack.stackTop -= reducedNTaken;
                Assert(stack.stackTop >= 0);

                enumerate_sub_csg_emit<BitmapsetN,STACK_SIZE>(
                                T, emit_S, emit_X, I, E, 
                                loop_stack, loop_stack_size,
                                jr_out, join_stack, memo, info);
            } else{
                int wScan = __popc(~pthBlt & LANE_MASK_LE);
                int pos = W_OFFSET + stack.stackTop + wScan - 1;
                if (p){
                    Assert(pos >= W_OFFSET && pos < W_OFFSET + WARP_SIZE);
                    stack.ctxStack[pos].S = emit_S;
                    stack.ctxStack[pos].X = emit_X;
                    stack.ctxStack[pos].I = I;
                    LOG_DEBUG("[%d: %d] Accumulating stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S.toUint(), emit_X.toUint(), I.toUint());
                }
                stack.stackTop += WARP_SIZE - reducedNTaken;
                Assert(stack.stackTop <= WARP_SIZE);
            }
        } else {
            if (LANE_ID == 0){
                LOG_DEBUG("[%d] Clearing stack (%d)\n", W_OFFSET, stack.stackTop);
            }
            if (LANE_ID < stack.stackTop){
                int pos = W_OFFSET + stack.stackTop - LANE_ID - 1;
                Assert(pos >= W_OFFSET && pos < W_OFFSET + WARP_SIZE);

                emit_S = stack.ctxStack[pos].S;
                emit_X = stack.ctxStack[pos].X;
                I = stack.ctxStack[pos].I;
                E = R - I;

                LOG_DEBUG("[%d: %d] Consuming stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S.toUint(), emit_X.toUint(), I.toUint());
            } else {
                // fake values just to get to try_join
                // these will fail all checks and make the thread execute any
                // pending join
                // this thread will not touch any queue

                emit_S = BitmapsetN(0);
                emit_X = T;
            }

            enumerate_sub_csg_emit<BitmapsetN,STACK_SIZE>(
                T, emit_S, emit_X, I, E, 
                loop_stack, loop_stack_size,
                jr_out, join_stack, memo, info);

            stack.stackTop = 0;
        }

        all_empty = __all_sync(WARP_MASK, loop_stack_size == 0);
    }
}

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH