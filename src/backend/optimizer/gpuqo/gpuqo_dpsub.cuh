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

template <typename stack_elem_t>
struct ccc_stack_t{
    volatile stack_elem_t* ctxStack;
    int stackTop;
};

template<typename BitmapsetN>
struct dpsub_iter_param_t{
    GpuqoPlannerInfo<BitmapsetN>* info;
    GpuqoPlannerInfo<BitmapsetN>* gpu_info;
    BitmapsetN out_relid;
    HashTableDpsub<BitmapsetN>* memo;
    thrust::host_vector<uint_t<BitmapsetN> > binoms;
    thrust::device_vector<uint_t<BitmapsetN> > gpu_binoms;
    uninit_device_vector<BitmapsetN> gpu_pending_keys;
    uninit_device_vector<BitmapsetN> gpu_scratchpad_keys;
    uninit_device_vector<JoinRelation<BitmapsetN> > gpu_scratchpad_vals;
    uninit_device_vector<BitmapsetN> gpu_reduced_keys;
    uninit_device_vector<JoinRelation<BitmapsetN> > gpu_reduced_vals;
    uint64_t n_sets;
    uint64_t n_joins_per_set;
    uint64_t tot;
    uint32_t scratchpad_size;
};

template<typename BitmapsetN>
using scatter_iter_t =  thrust::pair<typename uninit_device_vector<BitmapsetN>::iterator, typename uninit_device_vector<JoinRelation<BitmapsetN> >::iterator>;

template<typename BitmapsetN>
using pairs_enum_func_t = thrust::binary_function<BitmapsetN, uint_t<BitmapsetN>, JoinRelation<BitmapsetN> >;

template<typename BitmapsetN>
int dpsub_unfiltered_iteration(int iter, dpsub_iter_param_t<BitmapsetN> &params);

template<typename BitmapsetN>
int dpsub_filtered_iteration(int iter, dpsub_iter_param_t<BitmapsetN> &params);

template<typename BitmapsetN>
void dpsub_prune_scatter(int threads_per_set, int n_threads, 
                            dpsub_iter_param_t<BitmapsetN> &params);

template<typename BitmapsetN>
void dpsub_scatter(scatter_iter_t<BitmapsetN> scatter_from_iters, 
                scatter_iter_t<BitmapsetN> scatter_to_iters,
                dpsub_iter_param_t<BitmapsetN> &params);

template<typename BitmapsetN>
void dpsub_scatter(int n_sets, dpsub_iter_param_t<BitmapsetN> &params);

template<typename BitmapsetN>
using dpsub_filtered_evaluate_t = JoinRelation<BitmapsetN> (*)(
                                    BitmapsetN, uint32_t, int, 
                                    HashTableDpsub<BitmapsetN>&, 
                                    GpuqoPlannerInfo<BitmapsetN>*);

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
template<typename BitmapsetN>
__host__ __device__
__forceinline__
BitmapsetN dpsub_unrank_sid(uint_t<BitmapsetN> sid, uint32_t qss, uint32_t sq, uint_t<BitmapsetN>* binoms){
    BitmapsetN s = BitmapsetN::nth(sq).allLower();
    int qss_tmp = qss, sq_tmp = sq;

    while (sq_tmp > 0 && sq_tmp > qss_tmp){
        uint_t<BitmapsetN> o = BINOM(binoms, sq, sq_tmp-1, sq_tmp-qss_tmp-1);
        if (sid < o){
            s.unset(sq_tmp-1);
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
 template<typename BitmapsetN>
__host__ __device__
__forceinline__
BitmapsetN dpsub_unrank_next(BitmapsetN v){
    return v.nextPermutation();
}

template<typename BitmapsetN, bool CHECK_LEFT, bool CHECK_RIGHT>
__device__
__forceinline__
bool check_join(BitmapsetN left_id, BitmapsetN right_id, 
                GpuqoPlannerInfo<BitmapsetN>* info) {
    // make sure those subsets are valid
    if ((!CHECK_LEFT || is_connected(left_id, info->edge_table)) 
        && (!CHECK_RIGHT || is_connected(right_id, info->edge_table))
    ){
        // enumerator must generate disjoint sets
        Assert(is_disjoint(left_id, right_id));

        // enumerator must generate self-connected sets
        Assert(is_connected(left_id, info->edge_table));
        Assert(is_connected(right_id, info->edge_table));

        // We know that:
        //  join rel, left rel and right rel are connected;
        //  left_rel | right_rel = join_rel; left_rel & right_rel = 0 
        // Therefore left_rel must be connected to right_rel, otherwise
        //  join_rel would not be connected

        return true;
    } else {
        LOG_DEBUG("[bid:%d tid:%d] Invalid subsets %u and %u\n", 
                blockIdx.x, threadIdx.x, left_id.toUint(), right_id.toUint()
        );
        return false;
    }
}

template<typename BitmapsetN>
__device__
__forceinline__
void do_join(JoinRelation<BitmapsetN> &jr_out, 
             BitmapsetN left_rel_id, JoinRelation<BitmapsetN> &left_rel, 
             BitmapsetN right_rel_id, JoinRelation<BitmapsetN> &right_rel, 
             GpuqoPlannerInfo<BitmapsetN>* info) {
    LOG_DEBUG("[%3d,%3d] Joining %u and %u (%u)\n", 
                blockIdx.x, threadIdx.x,
                left_rel_id.toUint(), right_rel_id.toUint(),
                (left_rel_id | right_rel_id).toUint());

    Assert(!left_rel_id.empty());
    Assert(!right_rel_id.empty());

    float jr_rows = estimate_join_rows(left_rel_id, left_rel, right_rel_id, right_rel, info);
    float jr_cost = calc_join_cost(left_rel_id, left_rel, right_rel_id, right_rel, jr_rows, info);

    if (jr_cost < jr_out.cost){
        jr_out.cost = jr_cost;
        jr_out.rows = jr_rows;
        jr_out.left_rel_id = left_rel_id;
        jr_out.right_rel_id = right_rel_id;
    }
}

template<typename BitmapsetN, bool CHECK_LEFT, bool CHECK_RIGHT, bool NO_CCC>
__device__
void try_join(BitmapsetN jr, JoinRelation<BitmapsetN> &jr_out, 
                BitmapsetN l, BitmapsetN r, 
                bool additional_predicate, ccc_stack_t<BitmapsetN> &stack, 
                HashTableDpsub<BitmapsetN> &memo, 
                GpuqoPlannerInfo<BitmapsetN>* info)
{
    LOG_DEBUG("[%d, %d] try_join(%u, %u, %s)\n", 
                blockIdx.x, threadIdx.x, l.toUint(), r.toUint(),
                additional_predicate ? "true" : "false");

    bool p;
    p = additional_predicate && 
            check_join<BitmapsetN, CHECK_LEFT, CHECK_RIGHT>(l, r, info);

    Assert(__activemask() == WARP_MASK);

    if (NO_CCC){
        if (p){
            Assert(!l.empty() && !r.empty());

            JoinRelation<BitmapsetN> left_rel = *memo.lookup(l);
            JoinRelation<BitmapsetN> right_rel = *memo.lookup(r);
            do_join(jr_out, l, left_rel, r, right_rel, info);
        }
        return;
    }

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
            r = jr - l;
            LOG_DEBUG("[%d: %d] Consuming stack (%d=%d+%d-%d): l=%u, r=%u\n", 
                W_OFFSET, LANE_ID, pos, W_OFFSET, stack.stackTop, wScan, l.toUint(), r.toUint()
            );
        } else {
            LOG_DEBUG("[%d: %d] Using local values: l=%u, r=%u\n", 
                W_OFFSET, LANE_ID, l.toUint(), r.toUint()
            );
        }
        stack.stackTop -= reducedNTaken;

        Assert(!l.empty() && !r.empty());

        JoinRelation<BitmapsetN> left_rel = *memo.lookup(l);
        JoinRelation<BitmapsetN> right_rel = *memo.lookup(r);
        do_join(jr_out, l, left_rel, r, right_rel, info);

    } else{
        int wScan = __popc(~pthBlt & LANE_MASK_LE);
        int pos = W_OFFSET + stack.stackTop + wScan - 1;
        if (p){
            LOG_DEBUG("[%d: %d] Accumulating stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, l.toUint(), r.toUint());
            stack.ctxStack[pos] = l;
        }
        stack.stackTop += WARP_SIZE - reducedNTaken;
    }
    if (LANE_ID == 0){
        LOG_DEBUG("[%d] new stackTop=%d\n", W_OFFSET, stack.stackTop);
    }
}

#endif							/* GPUQO_DPSUB_CUH */
