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

#define PENDING_KEYS_SIZE(params) ((params).scratchpad_size*gpuqo_dpsub_filter_keys_overprovisioning)

#define BLOCK_DIM 256
#define WARP_SIZE 32
#define WARP_MASK 0xFFFFFFFF
#define LANE_ID (threadIdx.x & (WARP_SIZE-1))
#define WARP_ID (threadIdx.x & (~(WARP_SIZE-1)))
#define W_OFFSET WARP_ID
#define LANE_MASK_LE (WARP_MASK >> (WARP_SIZE-1-LANE_ID))

typedef struct join_stack_elem_t{
    JoinRelation *left_rel;
    JoinRelation *right_rel;
    // int padding;
} join_stack_elem_t;

template <typename stack_elem_t>
struct ccc_stack_t{
    volatile stack_elem_t* ctxStack;
    int stackTop;
};

typedef struct ccc_stack_t<join_stack_elem_t> join_stack_t;

typedef struct dpsub_iter_param_t{
    GpuqoPlannerInfo* info;
    RelationID out_relid;
    thrust::device_vector<JoinRelation> gpu_memo_vals;
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

__host__ __device__
__forceinline__
RelationID dpsub_unrank_sid(uint32_t sid, uint32_t qss, uint32_t sq, uint32_t* binoms){
    RelationID s = BMS32_EMPTY;
    int t = 0;
    int qss_tmp = qss, sq_tmp = sq;

    while (sq_tmp > 0 && qss_tmp > 0){
        uint32_t o = BINOM(binoms, sq, sq_tmp-1, qss_tmp-1);
        if (sid < o){
            s = BMS32_UNION(s, BMS32_NTH(t));
            qss_tmp--;
        } else {
            sid -= o;
        }
        t++;
        sq_tmp--;
    }

    return s;
}

template<bool CHECK_LEFT>
__device__
__forceinline__
bool check_join(RelationID left_id, JoinRelation &left_rel, 
                RelationID right_id, JoinRelation &right_rel, 
                GpuqoPlannerInfo* info) {
    // make sure those subsets were valid in a previous iteration
    if ((!CHECK_LEFT || left_rel.id != BMS32_EMPTY) && right_rel.id != BMS32_EMPTY){       
        // enumerator must generate disjoint sets
        Assert(is_disjoint(left_rel, right_rel));

        // enumerator must generate self-connected sets
        Assert(is_connected(left_rel.id, info->edge_table));
        Assert(is_connected(right_rel.id, info->edge_table));

        // left and right are inverted to continue accessing right relation 
        // in case left was not checked. Doing so may yield a cache hit.
        if (are_connected(right_rel.edges, left_id, info)){
            return true;
        } else {
            LOG_DEBUG("[%u] Cannot join %u and %u\n", BMS32_UNION(left_rel.id, right_rel.id), left_rel.id, right_rel.id);
            return false;
        }
    } else {
        LOG_DEBUG("[%u] Invalid subsets %u and %u\n", BMS32_UNION(left_rel.id, right_rel.id), left_rel.id, right_rel.id);
        return false;
    }
}

__device__
__forceinline__
void do_join(JoinRelation &jr_out, JoinRelation &left_rel, 
             JoinRelation &right_rel, GpuqoPlannerInfo* info) {
    LOG_DEBUG("[%u] Joining %u and %u\n", 
            relid, left_rel.id, right_rel.id);

    JoinRelation jr;
    make_join_rel(jr, left_rel, right_rel, info);

    if (jr.cost < jr_out.cost){
        jr_out = jr;
    }
}

template<bool CHECK_LEFT>
__device__
void try_join(JoinRelation &jr_out, RelationID l, RelationID r, 
            bool additional_predicate, join_stack_t &stack, 
            JoinRelation* memo_vals, GpuqoPlannerInfo* info);

#endif							/* GPUQO_DPSUB_CUH */
