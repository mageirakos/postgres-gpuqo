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

#define PENDING_KEYS_SIZE (gpuqo_dpsub_n_parallel*gpuqo_dpsub_filter_keys_overprovisioning)

#define WARP_SIZE 32
#define WARP_MASK 0xFFFFFFFF
#define BLOCK_DIM 256

typedef struct join_stack_elem_t{
    JoinRelation *left_rel;
    JoinRelation *right_rel;
    // int padding;
} join_stack_elem_t;

template <typename stack_elem_t>
struct ccc_stack_t{
    volatile stack_elem_t* ctxStack;
    int stackTop;
    int wOffset;
    int lane_id;
    unsigned lanemask_le;
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

__device__
__forceinline__
bool check_join(JoinRelation &left_rel, JoinRelation &right_rel, 
                GpuqoPlannerInfo* info) {
    // make sure those subsets were valid in a previous iteration
    if (left_rel.id != BMS32_EMPTY && right_rel.id != BMS32_EMPTY){       
        // enumerator must generate disjoint sets
        Assert(is_disjoint(left_rel, right_rel));

        // enumerator must generate self-connected sets
        Assert(is_connected(left_rel.id, info->edge_table));
        Assert(is_connected(right_rel.id, info->edge_table));

        if (are_connected(left_rel, right_rel, info)){
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
void do_join(RelationID relid, JoinRelation &jr_out, 
            JoinRelation &left_rel, JoinRelation &right_rel,
            GpuqoPlannerInfo* info) {
    LOG_DEBUG("[%u] Joining %u and %u\n", 
            relid, left_rel.id, right_rel.id);

    JoinRelation jr;
    jr.id = relid;
    jr.left_relation_id = left_rel.id;
    jr.left_relation_idx = left_rel.id;
    jr.right_relation_id = right_rel.id;
    jr.right_relation_idx = right_rel.id;
    jr.edges = BMS32_UNION(left_rel.edges, right_rel.edges);
    jr.rows = estimate_join_rows(jr, left_rel, right_rel, info);
    jr.cost = compute_join_cost(jr, left_rel, right_rel, info);

    if (jr.cost < jr_out.cost){
        jr_out = jr;
    }
}

__device__
void try_join(RelationID relid, JoinRelation &jr_out, 
            RelationID l, RelationID r, bool additional_predicate,
            join_stack_t &stack, JoinRelation* memo_vals, 
            GpuqoPlannerInfo* info);

#endif							/* GPUQO_DPSUB_CUH */
