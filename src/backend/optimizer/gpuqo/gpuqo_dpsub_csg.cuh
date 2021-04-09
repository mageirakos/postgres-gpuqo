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

typedef struct ext_loop_stack_elem_t{
    RelationID S;
    RelationID X;
    RelationID I;
    RelationID N;
} ext_loop_stack_elem_t;

typedef struct emit_stack_elem_t{
    RelationID S;
    RelationID X;
    RelationID I;
} emit_stack_elem_t;

typedef ccc_stack_t<emit_stack_elem_t> csg_stack_t;

template<int MAX_RELS, int STACK_SIZE>
__device__
static void enumerate_sub_csg(RelationID T, RelationID I, RelationID E,
                    JoinRelation &jr_out, join_stack_t &join_stack,
                    HashTable32bit &memo, GpuqoPlannerInfo* info);

template<int STACK_SIZE>
__device__ 
static void enumerate_sub_csg_emit(RelationID T, RelationID emit_S, 
            RelationID emit_X, RelationID I, RelationID E,
            ext_loop_stack_elem_t* loop_stack, int &loop_stack_size,
            JoinRelation &jr_out, join_stack_t &join_stack, 
            HashTable32bit &memo, GpuqoPlannerInfo* info);

__device__
static JoinRelation dpsubEnumerateCsg(RelationID relid, uint32_t cid, 
                                int n_splits, HashTable32bit &memo, 
                                GpuqoPlannerInfo* info)
{ 
    Assert(n_splits % 32 == 0 && BMS32_SIZE((Bitmapset32) n_splits) == 1);

    Bitmapset32 n_splits_bms = BMS32_HIGHEST((Bitmapset32) n_splits);
    Bitmapset32 cmp_cid = (n_splits_bms)-1 - cid;

    JoinRelation jr_out;
    jr_out.cost = INFD;

    volatile __shared__ join_stack_elem_t ctxStack[BLOCK_DIM];
    join_stack_t join_stack;
    join_stack.ctxStack = ctxStack;
    join_stack.stackTop = 0;

    Assert(BMS32_HIGHEST_POS(n_splits_bms)-1 <= BMS32_SIZE(relid));
    LOG_DEBUG("[%u, %u] n_splits_bms=%u, cmp_cid=%u\n", 
        relid, cid, n_splits_bms, cmp_cid);

    Assert(BMS32_UNION(cid, cmp_cid) == n_splits_bms-1);
    Assert(!BMS32_INTERSECTS(cid, cmp_cid));

    RelationID inc_set = BMS32_EXPAND_TO_MASK(cid, relid);
    RelationID exc_set = BMS32_EXPAND_TO_MASK(cmp_cid, relid);
    
    enumerate_sub_csg<32,64>(relid, inc_set, exc_set, jr_out, join_stack, 
        memo, info);

    if (LANE_ID < join_stack.stackTop){
        int pos = W_OFFSET + join_stack.stackTop - LANE_ID - 1;
        Assert(pos >= W_OFFSET && pos < W_OFFSET + WARP_SIZE);
        RelationID l = join_stack.ctxStack[pos];
        RelationID r = BMS32_DIFFERENCE(relid, l);

        LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", 
                W_OFFSET, LANE_ID, pos, l, r);

        JoinRelation left_rel = *memo.lookup(l);
        JoinRelation right_rel = *memo.lookup(r);
        do_join(jr_out, l, left_rel, r, right_rel, info);
    }
    
    return jr_out;
}

template<int STACK_SIZE>
__device__ 
static void enumerate_sub_csg_emit(RelationID T, RelationID emit_S, 
            RelationID emit_X, RelationID I, RelationID E,
            ext_loop_stack_elem_t* loop_stack, int &loop_stack_size,
            JoinRelation &jr_out, join_stack_t &join_stack, 
            HashTable32bit &memo, GpuqoPlannerInfo* info){
    // LOG_DEBUG("enumerate_sub_csg_emit(%u, %u, %u, %u, %u)\n",
    //         T, emit_S, emit_X, I, E);
    Assert(BMS32_IS_SUBSET(emit_S, T));
    Assert(BMS32_IS_SUBSET(I, T));
    Assert(emit_S == BMS32_EMPTY || is_connected(emit_S, info->edge_table));

    try_join<false,true>(T, jr_out, emit_S, BMS32_DIFFERENCE(T, emit_S), 
            emit_S != BMS32_EMPTY && BMS32_IS_SUBSET(I, emit_S), 
            join_stack, memo, info);

    RelationID new_N = BMS32_INTERSECTION(
        BMS32_DIFFERENCE(
            get_neighbours(emit_S, info->edge_table), 
            emit_X
        ),
        BMS32_DIFFERENCE(T, E)
    );
    
    // If possible, directly move to smaller I (it does not make 
    // sense to explore other rels in I first since it won't be 
    // possible to go back)
    RelationID lowI = BMS32_LOWEST(BMS32_DIFFERENCE(I, emit_S));
    if (BMS32_INTERSECTS(lowI, new_N)){
        new_N = lowI;
    }

    // do not add useless elements to stack
    if (new_N != BMS32_EMPTY){
        loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t){
            emit_S, emit_X, I, new_N
        };
        Assert(loop_stack_size < STACK_SIZE);
    }
}

template<int MAX_RELS, int STACK_SIZE>
__device__
static void enumerate_sub_csg(RelationID T, RelationID I, RelationID E,
                    JoinRelation &jr_out, join_stack_t &join_stack,
                    HashTable32bit &memo, GpuqoPlannerInfo* info){
    RelationID R = BMS32_UNION(I, E);

    Assert(BMS32_IS_SUBSET(I, T));

    // S, X, I, N
    ext_loop_stack_elem_t loop_stack[STACK_SIZE];
    int loop_stack_size = 0;

    volatile __shared__ emit_stack_elem_t ctxStack[BLOCK_DIM];
    csg_stack_t stack;
    stack.ctxStack = ctxStack;
    stack.stackTop = 0;

    LOG_DEBUG("[%d: %d] lanemask_le=%u\n", W_OFFSET, LANE_ID, LANE_MASK_LE);

    RelationID temp;
    if (I != BMS32_EMPTY){
        temp = BMS32_LOWEST(I);
    } else{
        temp = BMS32_DIFFERENCE(T, E);
    }

    while (temp != BMS32_EMPTY){
        RelationID v = BMS32_LOWEST(temp);
        
        loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t){
            BMS32_EMPTY,
            BMS32_SET_ALL_LOWER_INC(v),
            I,
            v
        };
        Assert(loop_stack_size < STACK_SIZE);
        temp = BMS32_DIFFERENCE(temp, v);
    }

    bool all_empty = false;
    while (!all_empty || stack.stackTop != 0){
        RelationID emit_S;
        RelationID emit_X;
        RelationID next_subset = BMS32_EMPTY;

        if (loop_stack_size != 0){
            ext_loop_stack_elem_t top = loop_stack[--loop_stack_size];
            RelationID S = top.S;
            RelationID X = top.X;
            RelationID N = top.N;
            I = top.I;
            E = BMS32_DIFFERENCE(R,I);

            Assert(loop_stack_size >= 0);

            // LOG_DEBUG("[%u: %u, %u] loop_stack: S=%u, X=%u, N=%u\n", T, I, E, S, X, N);

            next_subset = BMS32_NEXT_SUBSET(BMS32_INTERSECTION(S,N), N);

            emit_S = BMS32_UNION(BMS32_DIFFERENCE(S, N), next_subset);
            emit_X = BMS32_UNION(X, N);

            if (next_subset != BMS32_EMPTY){
                loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t){
                    emit_S, X, I, N
                };
                Assert(loop_stack_size < STACK_SIZE);
            }
        }

        if (!all_empty){
            bool p = next_subset != BMS32_EMPTY;
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
                    E = BMS32_DIFFERENCE(R,I);
                    LOG_DEBUG("[%d: %d] Consuming stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S, emit_X, I);
                }
                stack.stackTop -= reducedNTaken;
                Assert(stack.stackTop >= 0);

                enumerate_sub_csg_emit<STACK_SIZE>(T, emit_S, emit_X, I, E, 
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
                    LOG_DEBUG("[%d: %d] Accumulating stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S, emit_X, I);
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
                E = BMS32_DIFFERENCE(R,I);

                LOG_DEBUG("[%d: %d] Consuming stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S, emit_X, I);
            } else {
                // fake values just to get to try_join
                // these will fail all checks and make the thread execute any
                // pending join
                // this thread will not touch any queue

                emit_S = BMS32_EMPTY;
                emit_X = T;
            }

            enumerate_sub_csg_emit<STACK_SIZE>(T, emit_S, emit_X, I, E, 
                loop_stack, loop_stack_size,
                jr_out, join_stack, memo, info);

            stack.stackTop = 0;
        }

        all_empty = __all_sync(WARP_MASK, loop_stack_size == 0);
    }
}

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH