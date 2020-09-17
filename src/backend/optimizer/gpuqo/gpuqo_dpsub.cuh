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
bool check_join(JoinRelation &left_rel, JoinRelation &right_rel, 
                GpuqoPlannerInfo* info) {
    // make sure those subsets were valid in a previous iteration
    if ((!CHECK_LEFT || left_rel.id != BMS32_EMPTY) && right_rel.id != BMS32_EMPTY){       
        // enumerator must generate disjoint sets
        Assert(is_disjoint(left_rel.id, right_rel.id));

        // enumerator must generate self-connected sets
        if (CHECK_LEFT){
            Assert(is_connected(left_rel.id, info->edge_table));
        } else{ 
            // if not checking left, it might happen that it is 0 
            // but it's being taken care of in try_join
            Assert(left_rel.id == 0 || is_connected(left_rel.id, info->edge_table));
        }
        Assert(is_connected(right_rel.id, info->edge_table));

        // We know that:
        //  join rel, left rel and right rel are connected;
        //  left_rel | right_rel = join_rel; left_rel & right_rel = 0 
        // Therefore left_rel must be connected to right_rel, otherwise
        //  join_rel would not be connected
        // if left_rel.id == 0 then it is already taken care of so do 
        //  not trigger the assertion
        Assert((!CHECK_LEFT && left_rel.id == 0) || are_connected(left_rel, right_rel, info));

        return true;
    } else {
        LOG_DEBUG("[bid:%d tid:%d] Invalid subsets %u and %u\n", 
                blockIdx.x, threadIdx.x, left_rel.id, right_rel.id);
        return false;
    }
}

__device__
__forceinline__
void do_join(JoinRelation &jr_out, JoinRelation &left_rel, 
             JoinRelation &right_rel, GpuqoPlannerInfo* info) {
    LOG_DEBUG("[%u] Joining %u and %u\n", 
            BMS32_UNION(left_rel.id, right_rel.id), left_rel.id, right_rel.id);

    Assert(left_rel.id != BMS32_EMPTY);
    Assert(right_rel.id != BMS32_EMPTY);

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
