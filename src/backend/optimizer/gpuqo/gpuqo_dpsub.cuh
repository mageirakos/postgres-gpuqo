/*-------------------------------------------------------------------------
 *
 * gpuqo_dpsub.cuh
 *	  declaration of gpuqo DPsub related functions and functors.
 *
 * src/include/optimizer/gpuqo_dpsub.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_CUH
#define GPUQO_DPSUB_CUH

#include <thrust/tabulate.h>

#include "gpuqo.cuh"
#include "gpuqo_binomial.cuh"
#include "gpuqo_hashtable.cuh"

#define PENDING_KEYS_SIZE(params) ((params).scratchpad_size*gpuqo_dpsub_filter_keys_overprovisioning)

#define BLOCK_DIM 256
#define WARP_SIZE 32
#define WARP_MASK 0xFFFFFFFF
#define LANE_ID (threadIdx.x & (WARP_SIZE-1))
#define WARP_ID (threadIdx.x & (~(WARP_SIZE-1)))
#define W_OFFSET WARP_ID
#define LANE_MASK_LE (WARP_MASK >> (WARP_SIZE-1-LANE_ID))

typedef RelationID join_stack_elem_t;

template <typename stack_elem_t>
struct ccc_stack_t{
    volatile stack_elem_t* ctxStack;
    int stackTop;
};

typedef struct ccc_stack_t<join_stack_elem_t> join_stack_t;

typedef struct dpsub_iter_param_t{
    GpuqoPlannerInfo* info;
    RelationID out_relid;
    HashTable32bit* memo;
    thrust::host_vector<uint32_t> binoms;
    thrust::device_vector<uint32_t> gpu_binoms;
    uninit_device_vector_relid gpu_pending_keys;
    uninit_device_vector_relid gpu_scratchpad_keys;
    uninit_device_vector_joinrel gpu_scratchpad_vals;
    uninit_device_vector_relid gpu_reduced_keys;
    uninit_device_vector_joinrel gpu_reduced_vals;
    uint32_t n_sets;
    uint32_t n_joins_per_set;
    uint64_t tot;
    uint32_t scratchpad_size;
} dpsub_iter_param_t;

typedef thrust::pair<uninit_device_vector_relid::iterator, uninit_device_vector_joinrel::iterator> scatter_iter_t;

typedef thrust::binary_function<RelationID, uint32_t, JoinRelation> pairs_enum_func_t;

int dpsub_unfiltered_iteration(int iter, dpsub_iter_param_t &params);
int dpsub_filtered_iteration(int iter, dpsub_iter_param_t &params);

void dpsub_prune_scatter(int threads_per_set, int n_threads, dpsub_iter_param_t &params);

EXTERN_PROTOTYPE_TIMING(unrank);
EXTERN_PROTOTYPE_TIMING(filter);
EXTERN_PROTOTYPE_TIMING(compute);
EXTERN_PROTOTYPE_TIMING(prune);
EXTERN_PROTOTYPE_TIMING(scatter);
EXTERN_PROTOTYPE_TIMING(iteration);

/**
 * Unrank the id of the set (sid) to the corresponding set relationID
 *
 * Note: the unranked sets are in lexicographical order
 */
__host__ __device__
__forceinline__
RelationID dpsub_unrank_sid(uint32_t sid, uint32_t qss, uint32_t sq, uint32_t* binoms){
    RelationID s = BMS32_SET_ALL_LOWER(BMS32_NTH(sq));
    int qss_tmp = qss, sq_tmp = sq;

    while (sq_tmp > 0 && sq_tmp > qss_tmp){
        uint32_t o = BINOM(binoms, sq, sq_tmp-1, sq_tmp-qss_tmp-1);
        if (sid < o){
            s = BMS32_UNSET(s, sq_tmp-1);
        } else {
            qss_tmp--;
            sid -= o;
        }
        sq_tmp--;
    }

    return s;
}

/**
 * Compute the lexicographically next bit permutation
 *
 * https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
 */
__host__ __device__
__forceinline__
RelationID dpsub_unrank_next(RelationID v){
    unsigned int t = (v | (v - 1)) + 1;  
    return t | ((((t & -t) / (v & -v)) >> 1) - 1);  
}

template<bool CHECK_LEFT>
__device__
__forceinline__
bool check_join(RelationID left_id, RelationID right_id, 
                GpuqoPlannerInfo* info) {
    // make sure those subsets are valid
    if ((!CHECK_LEFT || is_connected(left_id, info->edge_table)) 
        && is_connected(right_id, info->edge_table)){       
        // enumerator must generate disjoint sets
        Assert(is_disjoint(left_id, right_id));

        // enumerator must generate self-connected sets
        Assert(!CHECK_LEFT || is_connected(left_id, info->edge_table));
        Assert(is_connected(right_id, info->edge_table));

        // We know that:
        //  join rel, left rel and right rel are connected;
        //  left_rel | right_rel = join_rel; left_rel & right_rel = 0 
        // Therefore left_rel must be connected to right_rel, otherwise
        //  join_rel would not be connected

        return true;
    } else {
        LOG_DEBUG("[bid:%d tid:%d] Invalid subsets %u and %u\n", 
                blockIdx.x, threadIdx.x, left_id, right_id
        );
        return false;
    }
}

__device__
__forceinline__
void do_join(JoinRelation &jr_out, 
             RelationID left_rel_id, JoinRelation &left_rel, 
             RelationID right_rel_id, JoinRelation &right_rel, 
             GpuqoPlannerInfo* info) {
    LOG_DEBUG("[%u] Joining %u and %u\n", 
            BMS32_UNION(left_rel_id, right_rel_id), left_rel_id, right_rel_id);

    Assert(left_rel_id != BMS32_EMPTY);
    Assert(right_rel_id != BMS32_EMPTY);

    float jr_rows = estimate_join_rows(left_rel_id, left_rel, right_rel_id, right_rel, info);
    float jr_cost = calc_join_cost(left_rel_id, left_rel, right_rel_id, right_rel, jr_rows, info);

    if (jr_cost < jr_out.cost){
        jr_out.cost = jr_cost;
        jr_out.rows = jr_rows;
        jr_out.left_rel_id = left_rel_id;
        jr_out.right_rel_id = right_rel_id;
    }
}

template<bool CHECK_LEFT>
__device__
void try_join(JoinRelation &jr_out, RelationID l, RelationID r, 
                bool additional_predicate, join_stack_t &stack, 
                HashTable32bit &memo, GpuqoPlannerInfo* info)
{
    LOG_DEBUG("[%d, %d] try_join(%u, %u, %s)\n", 
                blockIdx.x, threadIdx.x, l, r,
                additional_predicate ? "true" : "false");

    RelationID jr = BMS32_UNION(l, r);

    bool p = additional_predicate && check_join<CHECK_LEFT>(l, r, info);

    Assert(__activemask() == WARP_MASK);

    unsigned pthBlt = __ballot_sync(WARP_MASK, !p);
    int reducedNTaken = __popc(pthBlt);
    if (LANE_ID == 0){
        LOG_DEBUG("[%d] pthBlt=%u, reducedNTaken=%d, stackTop=%d\n", W_OFFSET, pthBlt, reducedNTaken, stack.stackTop);
    }
    if (stack.stackTop >= reducedNTaken){
        int wScan = __popc(pthBlt & LANE_MASK_LE);
        int pos = W_OFFSET + stack.stackTop - wScan;
        if (!p){
            l = stack.ctxStack[pos];
            r = BMS32_DIFFERENCE(jr, l);
            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", 
                W_OFFSET, LANE_ID, pos, l, r
            );
        } else {
            LOG_DEBUG("[%d: %d] Using local values: l=%u, r=%u\n", 
                W_OFFSET, LANE_ID, l, r
            );
        }
        stack.stackTop -= reducedNTaken;

        Assert(l != BMS32_EMPTY && r != BMS32_EMPTY);

        JoinRelation left_rel = *memo.lookup(l);
        JoinRelation right_rel = *memo.lookup(r);
        do_join(jr_out, l, left_rel, r, right_rel, info);

    } else{
        int wScan = __popc(~pthBlt & LANE_MASK_LE);
        int pos = W_OFFSET + stack.stackTop + wScan - 1;
        if (p){
            LOG_DEBUG("[%d: %d] Accumulating stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, l, r);
            stack.ctxStack[pos] = l;
        }
        stack.stackTop += WARP_SIZE - reducedNTaken;
    }
    if (LANE_ID == 0){
        LOG_DEBUG("[%d] new stackTop=%d\n", W_OFFSET, stack.stackTop);
    }
}

#endif							/* GPUQO_DPSUB_CUH */
