/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_csg.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_csg.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo_dpsub_csg.cuh"

#define WARP_SIZE 32

// user-defined variables
bool gpuqo_dpsub_csg_enable;
int gpuqo_dpsub_csg_threshold;

typedef struct loop_stack_elem_t{
    RelationID S;
    RelationID X;
    RelationID N;
} loop_stack_elem_t;

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
            JoinRelation* memo_vals, GpuqoPlannerInfo* info){
    // LOG_DEBUG("enumerate_sub_csg_emit(%llu, %llu, %llu, %llu, %llu)\n",
    //         T, emit_S, emit_X, I, E);
    Assert(BMS64_IS_SUBSET(emit_S, T));
    Assert(BMS64_IS_SUBSET(I, T));

    try_join(T, jr_out, emit_S, BMS64_DIFFERENCE(T, emit_S), 
            BMS64_IS_SUBSET(I, emit_S), join_stack, memo_vals, info);

    RelationID new_N = BMS64_INTERSECTION(
        BMS64_DIFFERENCE(
            get_neighbours(emit_S, info), 
            emit_X
        ),
        BMS64_DIFFERENCE(T, E)
    );
    
    // If possible, directly move to smaller I (it does not make 
    // sense to explore other rels in I first since it won't be 
    // possible to go back)
    RelationID lowI = BMS64_LOWEST(BMS64_DIFFERENCE(I, emit_S));
    if (BMS64_INTERSECTS(lowI, new_N)){
        new_N = lowI;
    }

    // do not add useless elements to stack
    if (new_N != BMS64_EMPTY){
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
    RelationID R = BMS64_UNION(I, E);

    Assert(BMS64_IS_SUBSET(I, T));

    // S, X, I, N
    ext_loop_stack_elem_t loop_stack[MAX_DEPTH];
    int loop_stack_size = 0;

    volatile __shared__ emit_stack_elem_t ctxStack[BLOCK_DIM];
    csg_stack_t stack;
    stack.ctxStack = ctxStack;
    stack.stackTop = 0;
    stack.wOffset = threadIdx.x & (~(WARP_SIZE-1));
    stack.lane_id = threadIdx.x & (WARP_SIZE-1);
    stack.lanemask_le = (1 << (stack.lane_id+1)) - 1;

    LOG_DEBUG("[%d: %d] lanemask_le=%u\n", stack.wOffset, stack.lane_id, stack.lanemask_le);

    RelationID temp;
    if (I != BMS64_EMPTY){
        temp = BMS64_LOWEST(I);
    } else{
        temp = BMS64_DIFFERENCE(T, E);
    }

    while (temp != BMS64_EMPTY){
        int idx = BMS64_HIGHEST_POS(temp)-1;
        RelationID v = BMS64_NTH(idx);
        
        loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t){
            BMS64_EMPTY,
            BMS64_SET_ALL_LOWER_INC(v),
            I,
            v
        };
        temp = BMS64_DIFFERENCE(temp, v);
    }

    bool all_empty = false;
    while (!all_empty || stack.stackTop != 0){
        RelationID emit_S;
        RelationID emit_X;
        RelationID next_subset = BMS64_EMPTY;

        if (loop_stack_size != 0){
            ext_loop_stack_elem_t top = loop_stack[--loop_stack_size];
            RelationID S = top.S;
            RelationID X = top.X;
            RelationID N = top.N;
            I = top.I;
            E = BMS64_DIFFERENCE(R,I);

            // LOG_DEBUG("[%llu: %llu, %llu] loop_stack: S=%llu, X=%llu, N=%llu\n", T, I, E, S, X, N);

            next_subset = BMS64_NEXT_SUBSET(BMS64_INTERSECTION(S,N), N);

            emit_S = BMS64_UNION(BMS64_DIFFERENCE(S, N), next_subset);
            emit_X = BMS64_UNION(X, N);

            if (next_subset != BMS64_EMPTY){
                loop_stack[loop_stack_size++] = (ext_loop_stack_elem_t){
                    emit_S, X, I, N
                };
            }
        }

        if (!all_empty){
            bool p = next_subset != BMS64_EMPTY;
            unsigned pthBlt = __ballot_sync(WARP_MASK, !p);
            int reducedNTaken = __popc(pthBlt);
            if (stack.lane_id == 0){
                LOG_DEBUG("[%d] pthBlt=%u, reducedNTaken=%d, stackTop=%d\n", stack.wOffset, pthBlt, reducedNTaken, stack.stackTop);
            }
            if (stack.stackTop >= reducedNTaken){
                int wScan = __popc(pthBlt & stack.lanemask_le);
                int pos = stack.wOffset + stack.stackTop - wScan;
                if (!p){
                    emit_S = stack.ctxStack[pos].S;
                    emit_X = stack.ctxStack[pos].X;
                    I = stack.ctxStack[pos].I;
                    E = BMS64_DIFFERENCE(R,I);
                    LOG_DEBUG("[%d: %d] Consuming stack (%d): S=%llu, X=%llu, I=%llu\n", stack.wOffset, stack.lane_id, pos, emit_S, emit_X, I);
                }
                stack.stackTop -= reducedNTaken;

                enumerate_sub_csg_emit(T, emit_S, emit_X, I, E, 
                                loop_stack, loop_stack_size,
                                jr_out, join_stack, memo_vals, info);
            } else{
                int wScan = __popc(~pthBlt & stack.lanemask_le);
                int pos = stack.wOffset + stack.stackTop + wScan - 1;
                if (p){
                    stack.ctxStack[pos].S = emit_S;
                    stack.ctxStack[pos].X = emit_X;
                    stack.ctxStack[pos].I = I;
                    LOG_DEBUG("[%d: %d] Accumulating stack (%d): S=%llu, X=%llu, I=%llu\n", stack.wOffset, stack.lane_id, pos, emit_S, emit_X, I);
                }
                stack.stackTop += WARP_SIZE - reducedNTaken;
            }
        } else {
            if (stack.lane_id == 0){
                LOG_DEBUG("[%d] Clearing stack (%d)\n", stack.wOffset, stack.stackTop);
            }
            if (stack.lane_id < stack.stackTop){
                int pos = stack.wOffset + stack.stackTop - stack.lane_id - 1;
                emit_S = stack.ctxStack[pos].S;
                emit_X = stack.ctxStack[pos].X;
                I = stack.ctxStack[pos].I;
                E = BMS64_DIFFERENCE(R,I);

                LOG_DEBUG("[%d: %d] Consuming stack (%d): S=%llu, X=%llu, I=%llu\n", stack.wOffset, stack.lane_id, pos, emit_S, emit_X, I);
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
                jr_out, join_stack, memo_vals, info);

            stack.stackTop = 0;
        }

        all_empty = __all_sync(WARP_MASK, loop_stack_size == 0);
    }
}

__device__
JoinRelation dpsubEnumerateCsg::operator()(RelationID relid, uint64_t cid)
{ 
    Assert(n_splits % 32 == 0 && BMS64_SIZE((uint64_t)n_splits) == 1);

    uint64_t n_splits_u64 = BMS64_HIGHEST((uint64_t)n_splits);
    uint64_t cmp_cid = (n_splits_u64)-1 - cid;

    JoinRelation jr_out;
    jr_out.id = BMS64_EMPTY;
    jr_out.cost = INFD;

    volatile __shared__ join_stack_elem_t ctxStack[BLOCK_DIM];
    join_stack_t join_stack;
    join_stack.ctxStack = ctxStack;
    join_stack.stackTop = 0;
    join_stack.wOffset = threadIdx.x & (~(WARP_SIZE-1));
    join_stack.lane_id = threadIdx.x & (WARP_SIZE-1);
    join_stack.lanemask_le = (1 << (join_stack.lane_id+1)) - 1;

    Assert(BMS64_HIGHEST_POS(n_splits_u64)-1 <= BMS64_SIZE(relid));
    LOG_DEBUG("[%llu, %llu] n_splits_u64=%llu, cmp_cid=%llu\n", 
        relid, cid, n_splits_u64, cmp_cid);

    Assert(BMS64_UNION(cid, cmp_cid) == n_splits_u64-1);
    Assert(!BMS64_INTERSECTS(cid, cmp_cid));

    RelationID inc_set = BMS64_EXPAND_TO_MASK(cid, relid);
    RelationID exc_set = BMS64_EXPAND_TO_MASK(cmp_cid, relid);
    
    enumerate_sub_csg<64>(relid, inc_set, exc_set, jr_out, join_stack,
        memo_vals.get(), info);

    if (join_stack.lane_id < join_stack.stackTop){
        int pos = join_stack.wOffset + join_stack.stackTop - join_stack.lane_id - 1;
        JoinRelation *left_rel = join_stack.ctxStack[pos].left_rel;
        JoinRelation *right_rel = join_stack.ctxStack[pos].right_rel;

        LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%llu, r=%llu\n", join_stack.wOffset, join_stack.lane_id, pos, left_rel->id, right_rel->id);

        do_join(relid, jr_out, *left_rel, *right_rel, info);
    }
    
    return jr_out;
}
