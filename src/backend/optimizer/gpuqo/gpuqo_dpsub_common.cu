/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub.cu
 *
 *-------------------------------------------------------------------------
 */

#include <iostream>
#include <cmath>
#include <cstdint>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/tabulate.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/system/system_error.h>
#include <thrust/distance.h>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_binomial.cuh"
#include "gpuqo_query_tree.cuh"
#include "gpuqo_dpsub.cuh"

// relsize depends on algorithm
#define RELSIZE (sizeof(JoinRelation))

PROTOTYPE_TIMING(unrank);
PROTOTYPE_TIMING(filter);
PROTOTYPE_TIMING(compute);
PROTOTYPE_TIMING(prune);
PROTOTYPE_TIMING(scatter);
PROTOTYPE_TIMING(iteration);

// User-configured option
int gpuqo_n_parallel;

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

        JoinRelation *left_rel = memo.lookup(l);
        JoinRelation *right_rel = memo.lookup(r);
        do_join(jr_out, *left_rel, *right_rel, info);

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
template __device__ void try_join<true>(JoinRelation &jr_out, RelationID l, RelationID r, bool additional_predicate, join_stack_t &stack,  HashTable32bit &memo, GpuqoPlannerInfo* info);
template __device__ void try_join<false>(JoinRelation &jr_out, RelationID l, RelationID r, bool additional_predicate, join_stack_t &stack,  HashTable32bit &memo, GpuqoPlannerInfo* info);

void dpsub_prune_scatter(int threads_per_set, int n_threads, dpsub_iter_param_t &params){
    // give possibility to user to interrupt
    CHECK_FOR_INTERRUPTS();

    scatter_iter_t scatter_from_iters;
    scatter_iter_t scatter_to_iters;

    if (threads_per_set != 1){
        START_TIMING(prune);
        scatter_from_iters = thrust::make_pair(
            params.gpu_reduced_keys.begin(),
            params.gpu_reduced_vals.begin()
        );
        // prune to intermediate memory
        scatter_to_iters = thrust::reduce_by_key(
            params.gpu_scratchpad_keys.begin(),
            params.gpu_scratchpad_keys.begin() + n_threads,
            params.gpu_scratchpad_vals.begin(),
            params.gpu_reduced_keys.begin(),
            params.gpu_reduced_vals.begin(),
            thrust::equal_to<RelationID>(),
            thrust::minimum<JoinRelation>()
        );
        STOP_TIMING(prune);
    } else{
        scatter_from_iters = thrust::make_pair(
            params.gpu_scratchpad_keys.begin(),
            params.gpu_scratchpad_vals.begin()
        );
        scatter_to_iters = thrust::make_pair(
            (params.gpu_scratchpad_keys.begin()+n_threads),
            (params.gpu_scratchpad_vals.begin()+n_threads)
        );
    }

    LOG_DEBUG("After reduce_by_key\n");
    DUMP_VECTOR(scatter_from_iters.first, scatter_to_iters.first);
    DUMP_VECTOR(scatter_from_iters.second, scatter_to_iters.second);

    START_TIMING(scatter);
    params.memo->insert(
        scatter_from_iters.first.base().get(),
        scatter_from_iters.second.base().get(),
        thrust::distance(
            scatter_from_iters.first,
            scatter_to_iters.first
        )
    );
    STOP_TIMING(scatter);
}

/* gpuqo_dpsub
 *
 *	 GPU query optimization using the DP size variant.
 */
extern "C"
QueryTree*
gpuqo_dpsub(GpuqoPlannerInfo* info)
{
    DECLARE_TIMING(gpuqo_dpsub);
    DECLARE_NV_TIMING(init);
    DECLARE_NV_TIMING(execute);
    
    START_TIMING(gpuqo_dpsub);
    START_TIMING(init);

    size_t min_memo_cap = (size_t) gpuqo_min_memo_size_mb * MB / RELSIZE;
    size_t max_memo_cap = (size_t) gpuqo_max_memo_size_mb * MB / RELSIZE;
    size_t req_memo_size = 1ULL<<(info->n_rels);

    size_t memo_cap = std::min(req_memo_size*2, min_memo_cap);

    dpsub_iter_param_t params;
    params.info = info;
    params.memo = new HashTable32bit(memo_cap, max_memo_cap);
    thrust::host_vector<RelationID> ini_memo_keys(info->n_rels+1);
    thrust::host_vector<JoinRelation> ini_memo_vals(info->n_rels+1);
    thrust::device_vector<RelationID> ini_memo_keys_gpu(info->n_rels+1);
    thrust::device_vector<JoinRelation> ini_memo_vals_gpu(info->n_rels+1);

    QueryTree* out = NULL;
    params.out_relid = BMS32_EMPTY;

    for(int i=0; i<info->n_rels; i++){
        JoinRelation t;
        t.id = info->base_rels[i].id;
        t.left_relation_idx = 0; 
        t.left_relation_id = 0; 
        t.right_relation_idx = 0; 
        t.right_relation_id = 0; 
        t.cost = baserel_cost(info->base_rels[i]); 
        t.rows = info->base_rels[i].rows; 
        t.edges = info->edge_table[i];
        ini_memo_keys[i] = info->base_rels[i].id;
        ini_memo_vals[i] = t;

        params.out_relid = BMS32_UNION(params.out_relid, info->base_rels[i].id);
    }
    
    // add dummy relation
    JoinRelation dummy_jr;
    dummy_jr.id = BMS32_EMPTY;
	dummy_jr.edges = BMS32_EMPTY;
	dummy_jr.left_relation_id = BMS32_EMPTY;
	dummy_jr.right_relation_id = BMS32_EMPTY;
    dummy_jr.left_relation_ptr = NULL;
    dummy_jr.right_relation_ptr = NULL;
    dummy_jr.rows = 0.0;
	dummy_jr.cost = 0.0;
    
    ini_memo_keys[info->n_rels] = 0;
    ini_memo_vals[info->n_rels] = dummy_jr;

    // transfer base relations to GPU
    ini_memo_keys_gpu = ini_memo_keys;
    ini_memo_vals_gpu = ini_memo_vals;

    params.memo->insert(
        thrust::raw_pointer_cast(ini_memo_keys_gpu.data()), 
        thrust::raw_pointer_cast(ini_memo_vals_gpu.data()),
        info->n_rels+1
    );

    int binoms_size = (info->n_rels+1)*(info->n_rels+1);
    params.binoms = thrust::host_vector<uint32_t>(binoms_size);
    precompute_binoms(params.binoms, info->n_rels);
    params.gpu_binoms = params.binoms;

    params.scratchpad_size = (
        (
            gpuqo_scratchpad_size_mb * MB
        ) / (
            sizeof(RelationID)*gpuqo_dpsub_filter_keys_overprovisioning + 
            (sizeof(RelationID) + sizeof(JoinRelation))
        )
    );  

    if (params.scratchpad_size < gpuqo_n_parallel)
        params.scratchpad_size = gpuqo_n_parallel;

    LOG_PROFILE("Using a scratchpad of size %u\n", params.scratchpad_size);

    params.gpu_pending_keys = uninit_device_vector_relid(PENDING_KEYS_SIZE(params));
    params.gpu_scratchpad_keys = uninit_device_vector_relid(params.scratchpad_size);
    params.gpu_scratchpad_vals = uninit_device_vector_joinrel(params.scratchpad_size);
    params.gpu_reduced_keys = uninit_device_vector_relid(params.scratchpad_size);
    params.gpu_reduced_vals = uninit_device_vector_joinrel(params.scratchpad_size);

    STOP_TIMING(init);

    DUMP_VECTOR(params.gpu_binoms.begin(), params.gpu_binoms.end());    

    START_TIMING(execute);
    try{ // catch any exception in thrust
        INIT_NV_TIMING(unrank);
        INIT_NV_TIMING(filter);
        INIT_NV_TIMING(compute);
        INIT_NV_TIMING(prune);
        INIT_NV_TIMING(scatter);
        INIT_NV_TIMING(iteration);
        DECLARE_NV_TIMING(build_qt);

        // iterate over the size of the resulting joinrel
        for(int i=2; i<=info->n_rels; i++){
            // give possibility to user to interrupt
            CHECK_FOR_INTERRUPTS();
            
            // calculate number of combinations of relations that make up 
            // a joinrel of size i
            params.n_sets = BINOM(params.binoms, info->n_rels, info->n_rels, i);
            params.n_joins_per_set = ((1U)<<i);
            params.tot = ((uint64_t)params.n_sets) * params.n_joins_per_set;

            // used only if profiling is enabled
            uint32_t n_iters __attribute__((unused));
            uint64_t filter_threshold = ((uint64_t)gpuqo_n_parallel) * gpuqo_dpsub_filter_threshold;
            uint64_t csg_threshold = ((uint64_t)gpuqo_n_parallel) * gpuqo_dpsub_csg_threshold;

            START_TIMING(iteration);
            if ((gpuqo_dpsub_filter_enable && params.tot > filter_threshold) 
                    || (gpuqo_dpsub_csg_enable && params.tot > csg_threshold)){
                LOG_PROFILE("\nStarting filtered iteration %d: %llu combinations\n", i, params.tot);

                n_iters = dpsub_filtered_iteration(i, params);
            } else {
                LOG_PROFILE("\nStarting unfiltered iteration %d: %llu combinations\n", i, params.tot);

                n_iters = dpsub_unfiltered_iteration(i, params);
            }
            STOP_TIMING(iteration);

            LOG_DEBUG("It took %d iterations\n", n_iters);
            PRINT_CHECKPOINT_TIMING(unrank);
            PRINT_CHECKPOINT_TIMING(filter);
            PRINT_CHECKPOINT_TIMING(compute);
            PRINT_CHECKPOINT_TIMING(prune);
            PRINT_CHECKPOINT_TIMING(scatter);
            PRINT_TIMING(iteration);
        } // dpsub loop: for i = 2..n_rels

        START_TIMING(build_qt);
            
        buildQueryTree(params.out_relid, *params.memo, &out);
    
        STOP_TIMING(build_qt);
    
        PRINT_TOTAL_TIMING(unrank);
        PRINT_TOTAL_TIMING(filter);
        PRINT_TOTAL_TIMING(compute);
        PRINT_TOTAL_TIMING(prune);
        PRINT_TOTAL_TIMING(scatter);
    } catch(thrust::system_error err){
        printf("Thrust %d: %s\n", err.code().value(), err.what());
    }

    STOP_TIMING(execute);
    STOP_TIMING(gpuqo_dpsub);

    PRINT_TIMING(gpuqo_dpsub);
    PRINT_TIMING(init);
    PRINT_TIMING(execute);

    params.memo->free();
    delete params.memo;

    return out;
}
