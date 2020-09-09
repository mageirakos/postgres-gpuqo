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
int gpuqo_dpsub_n_parallel;

__host__ __device__
RelationID dpsub_unrank_sid(uint64_t sid, uint64_t qss, uint64_t sq, uint64_t* binoms){
    RelationID s = BMS64_EMPTY;
    int t = 0;
    int qss_tmp = qss, sq_tmp = sq;

    while (sq_tmp > 0 && qss_tmp > 0){
        uint64_t o = BINOM(binoms, sq, sq_tmp-1, qss_tmp-1);
        if (sid < o){
            s = BMS64_UNION(s, BMS64_NTH(t));
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
bool check_join(JoinRelation &left_rel, JoinRelation &right_rel, 
                GpuqoPlannerInfo* info) {
    // make sure those subsets were valid in a previous iteration
    if (left_rel.id != BMS64_EMPTY && right_rel.id != BMS64_EMPTY){       
        // enumerator must generate disjoint sets
        Assert(is_disjoint(left_rel, right_rel));

        // enumerator must generate self-connected sets
        Assert(is_connected(left_rel.id, info));
        Assert(is_connected(right_rel.id, info));

        if (are_connected(left_rel, right_rel, info)){
            return true;
        } else {
            LOG_DEBUG("[%llu] Cannot join %llu and %llu\n", BMS64_UNION(left_rel.id, right_rel.id), left_rel.id, right_rel.id);
            return false;
        }
    } else {
        LOG_DEBUG("[%llu] Invalid subsets %llu and %llu\n", BMS64_UNION(left_rel.id, right_rel.id), left_rel.id, right_rel.id);
        return false;
    }
}

__device__
void do_join(RelationID relid, JoinRelation &jr_out, 
            JoinRelation &left_rel, JoinRelation &right_rel,
            GpuqoPlannerInfo* info) {
    LOG_DEBUG("[%llu] Joining %llu and %llu\n", 
            relid, left_rel.id, right_rel.id);

    JoinRelation jr;
    jr.id = relid;
    jr.left_relation_id = left_rel.id;
    jr.left_relation_idx = left_rel.id;
    jr.right_relation_id = right_rel.id;
    jr.right_relation_idx = right_rel.id;
    jr.edges = BMS64_UNION(left_rel.edges, right_rel.edges);
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
            GpuqoPlannerInfo* info) {
    LOG_DEBUG("[%d, %d] try_join(%llu, %llu, %llu, %s)\n", 
                blockIdx.x, threadIdx.x, relid, l, r,
                additional_predicate ? "true" : "false");

    JoinRelation *left_rel = &memo_vals[l];
    JoinRelation *right_rel = &memo_vals[r];

    Assert(__activemask() == WARP_MASK);
    Assert(left_rel->id == BMS64_EMPTY || left_rel->id == l);
    Assert(right_rel->id == BMS64_EMPTY || right_rel->id == r);

    bool p = check_join(*left_rel, *right_rel, info) && additional_predicate;
    unsigned pthBlt = __ballot_sync(WARP_MASK, !p);
    int reducedNTaken = __popc(pthBlt);
    if (stack.lane_id == 0){
        LOG_DEBUG("[%d] pthBlt=%u, reducedNTaken=%d, stackTop=%d\n", stack.wOffset, pthBlt, reducedNTaken, stack.stackTop);
    }
    if (stack.stackTop >= reducedNTaken){
        int wScan = __popc(pthBlt & stack.lanemask_le);
        int pos = stack.wOffset + stack.stackTop - wScan;
        if (!p){
            left_rel = stack.ctxStack[pos].left_rel;
            right_rel = stack.ctxStack[pos].right_rel;
            LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%llu, r=%llu\n", stack.wOffset, stack.lane_id, pos, left_rel->id, right_rel->id);
        } else {
            LOG_DEBUG("[%d: %d] Using local values: l=%llu, r=%llu\n", stack.wOffset, stack.lane_id, left_rel->id, right_rel->id);
        }
        stack.stackTop -= reducedNTaken;

        Assert(BMS64_UNION(left_rel->id, right_rel->id) == relid);

        do_join(relid, jr_out, *left_rel, *right_rel, info);

    } else{
        int wScan = __popc(~pthBlt & stack.lanemask_le);
        int pos = stack.wOffset + stack.stackTop + wScan - 1;
        if (p){
            LOG_DEBUG("[%d: %d] Accumulating stack (%d): l=%llu, r=%llu\n", stack.wOffset, stack.lane_id, pos, left_rel->id, right_rel->id);
            stack.ctxStack[pos].left_rel = left_rel;
            stack.ctxStack[pos].right_rel = right_rel;
        }
        stack.stackTop += WARP_SIZE - reducedNTaken;
    }
    if (stack.lane_id == 0){
        LOG_DEBUG("[%d] new stackTop=%d\n", stack.wOffset, stack.stackTop);
    }
}

__device__
JoinRelation dpsubEnumerateAllSubs::operator()(RelationID relid, uint64_t cid)
{
    JoinRelation jr_out;
    jr_out.id = BMS64_EMPTY;
    jr_out.cost = INFD;
    int qss = BMS64_SIZE(relid);
    uint64_t n_possible_joins = 1ULL<<qss;
    uint64_t n_pairs = ceil_div(n_possible_joins, n_splits);
    uint64_t join_id = (cid)*n_pairs;
    RelationID l = BMS64_EXPAND_TO_MASK(join_id, relid);
    RelationID r;

    LOG_DEBUG("[%llu, %llu] n_splits=%d\n", relid, cid, n_splits);

    Assert(blockDim.x == BLOCK_DIM);
    volatile __shared__ join_stack_elem_t ctxStack[BLOCK_DIM];
    join_stack_t stack;
    stack.ctxStack = ctxStack;
    stack.stackTop = 0;
    stack.wOffset = threadIdx.x & (~(WARP_SIZE-1));
    stack.lane_id = threadIdx.x & (WARP_SIZE-1);
    stack.lanemask_le = (1 << (stack.lane_id+1)) - 1;

    bool stop = false;
    for (int i = 0; i < n_pairs; i++){
        stop = stop || (join_id+i != 0 && l == 0) || (join_id+i > n_possible_joins);

        if (stop){
            r=0; 
            // makes try_join process an invalid pair, giving it the possibility
            // to pop an element from the stack 
        } else {
            r = BMS64_DIFFERENCE(relid, l);
        }
        
        try_join(relid, jr_out, l, r, true, stack, memo_vals.get(), info);

        l = BMS64_NEXT_SUBSET(l, relid);
    }

    if (stack.lane_id < stack.stackTop){
        int pos = stack.wOffset + stack.stackTop - stack.lane_id - 1;
        JoinRelation *left_rel = stack.ctxStack[pos].left_rel;
        JoinRelation *right_rel = stack.ctxStack[pos].right_rel;

        LOG_DEBUG("[%d: %d] Consuming stack (%d): l=%llu, r=%llu\n", stack.wOffset, stack.lane_id, pos, left_rel->id, right_rel->id);

        do_join(relid, jr_out, *left_rel, *right_rel, info);
    }

    return jr_out;
}

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
            thrust::equal_to<uint64_t>(),
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

    uint64_t max_memo_size = gpuqo_dpsize_max_memo_size_mb * MB / RELSIZE;
    uint64_t req_memo_size = 1ULL<<(info->n_rels+1);
    if (max_memo_size < req_memo_size){
        printf("Insufficient memo size\n");
        return NULL;
    }

    uint64_t memo_size = std::min(req_memo_size, max_memo_size);

    dpsub_iter_param_t params;
    params.info = info;
    params.gpu_memo_vals = thrust::device_vector<JoinRelation>(memo_size);

    QueryTree* out = NULL;
    params.out_relid = BMS64_EMPTY;

    for(int i=0; i<info->n_rels; i++){
        JoinRelation t;
        t.id = info->base_rels[i].id;
        t.left_relation_idx = 0; 
        t.left_relation_id = 0; 
        t.right_relation_idx = 0; 
        t.right_relation_id = 0; 
        t.cost = baserel_cost(info->base_rels[i]); 
        t.rows = info->base_rels[i].rows; 
        t.edges = info->base_rels[i].edges;
        params.gpu_memo_vals[info->base_rels[i].id] = t;

        params.out_relid = BMS64_UNION(params.out_relid, info->base_rels[i].id);
    }

    int binoms_size = (info->n_rels+1)*(info->n_rels+1);
    params.binoms = thrust::host_vector<uint64_t>(binoms_size);
    precompute_binoms(params.binoms, info->n_rels);
    params.gpu_binoms = params.binoms;

    // scratchpad size is increased on demand, starting from a minimum capacity
    params.gpu_pending_keys = uninit_device_vector_relid(PENDING_KEYS_SIZE);
    params.gpu_scratchpad_keys = uninit_device_vector_relid(gpuqo_dpsub_n_parallel);
    params.gpu_scratchpad_vals = uninit_device_vector_joinrel(gpuqo_dpsub_n_parallel);
    params.gpu_reduced_keys = uninit_device_vector_relid(gpuqo_dpsub_n_parallel);
    params.gpu_reduced_vals = uninit_device_vector_joinrel(gpuqo_dpsub_n_parallel);

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
            params.n_joins_per_set = ((1ULL)<<i);
            params.tot = params.n_sets * params.n_joins_per_set;

            uint64_t n_iters;
            uint64_t filter_threshold = gpuqo_dpsub_n_parallel * gpuqo_dpsub_filter_threshold;
            uint64_t csg_threshold = gpuqo_dpsub_n_parallel * gpuqo_dpsub_csg_threshold;

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
