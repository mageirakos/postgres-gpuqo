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
                JoinRelation* memo_vals, GpuqoPlannerInfo* info)
{
    LOG_DEBUG("[%d, %d] try_join(%u, %u, %s)\n", 
                blockIdx.x, threadIdx.x, l, r,
                additional_predicate ? "true" : "false");

    JoinRelation *left_rel = &memo_vals[l];
    JoinRelation *right_rel = &memo_vals[r];

    Assert(__activemask() == WARP_MASK);
    Assert(left_rel->id == BMS32_EMPTY || left_rel->id == l);
    Assert(right_rel->id == BMS32_EMPTY || right_rel->id == r);

    bool p = additional_predicate && check_join<CHECK_LEFT>(l, *left_rel, r, *right_rel, info);
    unsigned pthBlt = __ballot_sync(WARP_MASK, !p);
    int reducedNTaken = __popc(pthBlt);
    if (LANE_ID == 0){
        LOG_DEBUG("[%d] pthBlt=%u, reducedNTaken=%d, stackTop=%d\n", W_OFFSET, pthBlt, reducedNTaken, stack.stackTop);
    }
    if (stack.stackTop >= reducedNTaken){
        int wScan = __popc(pthBlt & LANE_MASK_LE);
        int pos = W_OFFSET + stack.stackTop - wScan;
        if (!p){
            left_rel = stack.ctxStack[pos].left_rel;
            right_rel = stack.ctxStack[pos].right_rel;
            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, left_rel->id, right_rel->id);
        } else {
            LOG_DEBUG("[%d: %d] Using local values: l=%u, r=%u\n", W_OFFSET, LANE_ID, left_rel->id, right_rel->id);
        }
        stack.stackTop -= reducedNTaken;

        do_join(jr_out, *left_rel, *right_rel, info);

    } else{
        int wScan = __popc(~pthBlt & LANE_MASK_LE);
        int pos = W_OFFSET + stack.stackTop + wScan - 1;
        if (p){
            LOG_DEBUG("[%d: %d] Accumulating stack (%d): l=%u, r=%u\n", W_OFFSET, LANE_ID, pos, left_rel->id, right_rel->id);
            stack.ctxStack[pos].left_rel = left_rel;
            stack.ctxStack[pos].right_rel = right_rel;
        }
        stack.stackTop += WARP_SIZE - reducedNTaken;
    }
    if (LANE_ID == 0){
        LOG_DEBUG("[%d] new stackTop=%d\n", W_OFFSET, stack.stackTop);
    }
}
template __device__ void try_join<true>(JoinRelation &jr_out, RelationID l, RelationID r, bool additional_predicate, join_stack_t &stack,  JoinRelation* memo_vals, GpuqoPlannerInfo* info);
template __device__ void try_join<false>(JoinRelation &jr_out, RelationID l, RelationID r, bool additional_predicate, join_stack_t &stack,  JoinRelation* memo_vals, GpuqoPlannerInfo* info);

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
    thrust::scatter(
        scatter_from_iters.second,
        scatter_to_iters.second,
        scatter_from_iters.first,
        params.gpu_memo_vals.begin()
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

    uint32_t max_memo_size = gpuqo_max_memo_size_mb * MB / RELSIZE;
    uint32_t req_memo_size = 1U<<(info->n_rels+1);
    if (max_memo_size < req_memo_size){
        printf("Insufficient memo size\n");
        return NULL;
    }

    uint32_t memo_size = std::min(req_memo_size, max_memo_size);

    dpsub_iter_param_t params;
    params.info = info;
    params.gpu_memo_vals = thrust::device_vector<JoinRelation>(memo_size);

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
        params.gpu_memo_vals[info->base_rels[i].id] = t;

        params.out_relid = BMS32_UNION(params.out_relid, info->base_rels[i].id);
    }

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
    params.gpu_reduced_keys = uninit_device_vector_relid(params.scratchpad_size/32);
    params.gpu_reduced_vals = uninit_device_vector_joinrel(params.scratchpad_size/32);

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
            
        buildQueryTree(params.out_relid, params.gpu_memo_vals, &out);
    
        STOP_TIMING(build_qt);
    
        PRINT_TOTAL_TIMING(unrank);
        PRINT_TOTAL_TIMING(filter);
        PRINT_TOTAL_TIMING(compute);
        PRINT_TOTAL_TIMING(prune);
        PRINT_TOTAL_TIMING(scatter);
    } catch(thrust::system_error err){
        printf("Thrust %d: %s", err.code().value(), err.what());
    }

    STOP_TIMING(execute);
    STOP_TIMING(gpuqo_dpsub);

    PRINT_TIMING(gpuqo_dpsub);
    PRINT_TIMING(init);
    PRINT_TIMING(execute);

    return out;
}
