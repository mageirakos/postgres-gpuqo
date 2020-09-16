/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_csg.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_csg.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo_dpsub_csg.cuh"

// user-defined variables
bool gpuqo_dpsub_csg_enable;
int gpuqo_dpsub_csg_threshold;

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

__forceinline__ __device__ 
void enumerate_sub_csg_emit(RelationID T, RelationID emit_S, 
            RelationID emit_X, RelationID I, RelationID E,
            ext_loop_stack_elem_t* loop_stack, int &loop_stack_size,
            JoinRelation &jr_out, join_stack_t &join_stack, 
            JoinRelation* memo_vals, GpuqoPlannerInfo* info, EdgeMask* edge_table){
    // LOG_DEBUG("enumerate_sub_csg_emit(%u, %u, %u, %u, %u)\n",
    //         T, emit_S, emit_X, I, E);
    Assert(BMS32_IS_SUBSET(emit_S, T));
    Assert(BMS32_IS_SUBSET(I, T));

    try_join<false>(jr_out, emit_S, BMS32_DIFFERENCE(T, emit_S), 
            BMS32_IS_SUBSET(I, emit_S), join_stack, memo_vals, info);

    RelationID new_N = BMS32_INTERSECTION(
        BMS32_DIFFERENCE(
            get_neighbours(emit_S, edge_table), 
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
    }
}

template<int MAX_DEPTH>
__device__
void enumerate_sub_csg(RelationID T, RelationID I, RelationID E,
                    JoinRelation &jr_out, join_stack_t &join_stack,
                    JoinRelation* memo_vals, GpuqoPlannerInfo* info){
    RelationID R = BMS32_UNION(I, E);

    Assert(BMS32_IS_SUBSET(I, T));

    // S, X, I, N
    ext_loop_stack_elem_t loop_stack[MAX_DEPTH];
    int loop_stack_size = 0;

    volatile __shared__ emit_stack_elem_t ctxStack[BLOCK_DIM];
    csg_stack_t stack;
    stack.ctxStack = ctxStack;
    stack.stackTop = 0;

    LOG_DEBUG("[%d: %d] lanemask_le=%u\n", W_OFFSET, LANE_ID, LANE_MASK_LE);

    __shared__ EdgeMask edge_table[MAX_DEPTH];
    for (int i = threadIdx.x; i < info->n_rels; i+=blockDim.x){
        edge_table[i] = info->edge_table[i];
    }
    __syncthreads();

    RelationID temp;
    if (I != BMS32_EMPTY){
        temp = BMS32_LOWEST(I);
    } else{
        temp = BMS32_DIFFERENCE(T, E);
    }

    while (temp != BMS32_EMPTY){
        int idx = BMS32_HIGHEST_POS(temp)-1;
        RelationID v = BMS32_NTH(idx);
        
        loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t){
            BMS32_EMPTY,
            BMS32_SET_ALL_LOWER_INC(v),
            I,
            v
        };
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

            // LOG_DEBUG("[%u: %u, %u] loop_stack: S=%u, X=%u, N=%u\n", T, I, E, S, X, N);

            next_subset = BMS32_NEXT_SUBSET(BMS32_INTERSECTION(S,N), N);

            emit_S = BMS32_UNION(BMS32_DIFFERENCE(S, N), next_subset);
            emit_X = BMS32_UNION(X, N);

            if (next_subset != BMS32_EMPTY){
                loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t){
                    emit_S, X, I, N
                };
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
                    emit_S = stack.ctxStack[pos].S;
                    emit_X = stack.ctxStack[pos].X;
                    I = stack.ctxStack[pos].I;
                    E = BMS32_DIFFERENCE(R,I);
                    LOG_DEBUG("[%d: %d] Consuming stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S, emit_X, I);
                }
                stack.stackTop -= reducedNTaken;

                enumerate_sub_csg_emit(T, emit_S, emit_X, I, E, 
                                loop_stack, loop_stack_size,
                                jr_out, join_stack, memo_vals, info,
                                edge_table);
            } else{
                int wScan = __popc(~pthBlt & LANE_MASK_LE);
                int pos = W_OFFSET + stack.stackTop + wScan - 1;
                if (p){
                    stack.ctxStack[pos].S = emit_S;
                    stack.ctxStack[pos].X = emit_X;
                    stack.ctxStack[pos].I = I;
                    LOG_DEBUG("[%d: %d] Accumulating stack (%d): S=%u, X=%u, I=%u\n", W_OFFSET, LANE_ID, pos, emit_S, emit_X, I);
                }
                stack.stackTop += WARP_SIZE - reducedNTaken;
            }
        } else {
            if (LANE_ID == 0){
                LOG_DEBUG("[%d] Clearing stack (%d)\n", W_OFFSET, stack.stackTop);
            }
            if (LANE_ID < stack.stackTop){
                int pos = W_OFFSET + stack.stackTop - LANE_ID - 1;
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

                emit_S = E;
                emit_X = T;
            }

            enumerate_sub_csg_emit(T, emit_S, emit_X, I, E, 
                loop_stack, loop_stack_size,
                jr_out, join_stack, memo_vals, info, edge_table);

            stack.stackTop = 0;
        }

        all_empty = __all_sync(WARP_MASK, loop_stack_size == 0);
    }
}

__device__
JoinRelation dpsubEnumerateCsg::operator()(RelationID relid, uint32_t cid)
{ 
    Assert(n_splits % 32 == 0 && BMS32_SIZE((Bitmapset32) n_splits) == 1);

    Bitmapset32 n_splits_bms = BMS32_HIGHEST((Bitmapset32) n_splits);
    Bitmapset32 cmp_cid = (n_splits_bms)-1 - cid;

    JoinRelation jr_out;
    jr_out.id = BMS32_EMPTY;
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
    
    enumerate_sub_csg<32>(relid, inc_set, exc_set, jr_out, join_stack,
        memo_vals.get(), info);

    if (LANE_ID < join_stack.stackTop){
        int pos = W_OFFSET + join_stack.stackTop - LANE_ID - 1;
        JoinRelation *left_rel = join_stack.ctxStack[pos].left_rel;
        JoinRelation *right_rel = join_stack.ctxStack[pos].right_rel;

        LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, left_rel->id, right_rel->id);

        do_join(jr_out, *left_rel, *right_rel, info);
    }
    
    return jr_out;
}
