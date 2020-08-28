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

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"
#include "optimizer/gpuqo_binomial.cuh"
#include "optimizer/gpuqo_query_tree.cuh"
#include "optimizer/gpuqo_dpsub.cuh"

// relsize depends on algorithm
#define RELSIZE (sizeof(JoinRelation))

PROTOTYPE_TIMING(unrank);
PROTOTYPE_TIMING(filter);
PROTOTYPE_TIMING(compute);
PROTOTYPE_TIMING(prune);
PROTOTYPE_TIMING(scatter);

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
void try_join(RelationID relid, JoinRelation &jr_out, 
            RelationID l, RelationID r, JoinRelation* memo_vals,
            BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table) {
    if (l == BMS64_EMPTY || r == BMS64_EMPTY){
        return;
    }

#ifdef GPUQO_DEBUG
    printf("try_join(%llu, %llu, %llu)\n", relid, l, r);
#endif

    JoinRelation jr;
    jr.id = relid;
    jr.left_relation_id = l;
    jr.left_relation_idx = l;
    jr.right_relation_id = r;
    jr.right_relation_idx = r;
    
    JoinRelation left_rel = memo_vals[jr.left_relation_idx];
    JoinRelation right_rel = memo_vals[jr.right_relation_idx];

    // make sure those subsets were valid in a previous iteration
    if (left_rel.id == l && right_rel.id == r){
        jr.edges = BMS64_UNION(left_rel.edges, right_rel.edges);
        
        if (are_connected(left_rel, right_rel, base_rels, n_rels, edge_table)){

#ifdef GPUQO_DEBUG 
        printf("[%llu] Joining %llu and %llu\n", relid, l, r);
#endif

            jr.rows = estimate_join_rows(jr, left_rel, right_rel,
                                base_rels, n_rels, edge_table);

            jr.cost = compute_join_cost(jr, left_rel, right_rel,
                                base_rels, n_rels, edge_table);

            if (jr.cost < jr_out.cost){
                jr_out = jr;
            }
        } else {
#ifdef GPUQO_DEBUG 
            printf("[%llu] Cannot join %llu and %llu\n", relid, l, r);
#endif
        }
    } else {
#ifdef GPUQO_DEBUG 
        printf("[%llu] Invalid subsets %llu and %llu\n", relid, l, r);
#endif
    }
}

__device__
JoinRelation dpsubEnumerateAllSubs::operator()(RelationID relid, uint64_t cid)
{
    JoinRelation jr_out;
    jr_out.id = BMS64_EMPTY;
    jr_out.cost = INFD;
    RelationID l = BMS64_EXPAND_TO_MASK((cid)*n_pairs+1, relid);
    RelationID r;

    for (int i = 0; i < n_pairs; i++){
        r = BMS64_DIFFERENCE(relid, l);
        
        try_join(relid, jr_out, l, r, 
                memo_vals.get(), base_rels.get(), sq, edge_table.get());

        l = BMS64_NEXT_SUBSET(l, relid);
    }

    return jr_out;
}

void dpsub_prune_scatter(int n_joins_per_thread, int n_threads, dpsub_iter_param_t &params){
    // give possibility to user to interrupt
    CHECK_FOR_INTERRUPTS();

    scatter_iter_t scatter_from_iters;
    scatter_iter_t scatter_to_iters;

    if (n_joins_per_thread < params.n_joins_per_set){
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

#ifdef GPUQO_DEBUG
    printf("After reduce_by_key\n");
    printVector(scatter_from_iters.first, scatter_to_iters.first);
    printVector(scatter_from_iters.second, scatter_to_iters.second);
#endif

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
gpuqo_dpsub(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[])
{
    DECLARE_TIMING(gpuqo_dpsub);
    DECLARE_NV_TIMING(init);
    DECLARE_NV_TIMING(execute);
    
    START_TIMING(gpuqo_dpsub);
    START_TIMING(init);

    uint64_t max_memo_size = gpuqo_dpsize_max_memo_size_mb * MB / RELSIZE;
    uint64_t req_memo_size = 1ULL<<(n_rels+1);
    if (max_memo_size < req_memo_size){
        printf("Insufficient memo size\n");
        return NULL;
    }

    uint64_t memo_size = std::min(req_memo_size, max_memo_size);

    dpsub_iter_param_t params;
    params.base_rels = base_rels;
    params.n_rels = n_rels;
    params.edge_table = edge_table;
    
    params.gpu_base_rels = thrust::device_vector<BaseRelation>(base_rels, base_rels + n_rels);
    params.gpu_edge_table = thrust::device_vector<EdgeInfo>(edge_table, edge_table + n_rels*n_rels);
    params.gpu_memo_vals = thrust::device_vector<JoinRelation>(memo_size);

    QueryTree* out = NULL;
    params.out_relid = BMS64_EMPTY;

    for(int i=0; i<n_rels; i++){
        JoinRelation t;
        t.id = base_rels[i].id;
        t.left_relation_idx = 0; 
        t.left_relation_id = 0; 
        t.right_relation_idx = 0; 
        t.right_relation_id = 0; 
        t.cost = baserel_cost(base_rels[i]); 
        t.rows = base_rels[i].rows; 
        t.edges = base_rels[i].edges;
        params.gpu_memo_vals[base_rels[i].id] = t;

        params.out_relid = BMS64_UNION(params.out_relid, base_rels[i].id);
    }

    int binoms_size = (n_rels+1)*(n_rels+1);
    params.binoms = thrust::host_vector<uint64_t>(binoms_size);
    precompute_binoms(params.binoms, n_rels);
    params.gpu_binoms = params.binoms;

    // scratchpad size is increased on demand, starting from a minimum capacity
    params.gpu_pending_keys = uninit_device_vector_relid(PENDING_KEYS_SIZE);
    params.gpu_scratchpad_keys = uninit_device_vector_relid(gpuqo_dpsub_n_parallel);
    params.gpu_scratchpad_vals = uninit_device_vector_joinrel(gpuqo_dpsub_n_parallel);
    params.gpu_reduced_keys = uninit_device_vector_relid(gpuqo_dpsub_n_parallel);
    params.gpu_reduced_vals = uninit_device_vector_joinrel(gpuqo_dpsub_n_parallel);

    STOP_TIMING(init);

#ifdef GPUQO_DEBUG
    printVector(params.gpu_binoms.begin(), params.gpu_binoms.end());    
#endif

    START_TIMING(execute);
    try{ // catch any exception in thrust
        INIT_NV_TIMING(unrank);
        INIT_NV_TIMING(filter);
        INIT_NV_TIMING(compute);
        INIT_NV_TIMING(prune);
        INIT_NV_TIMING(scatter);
        DECLARE_NV_TIMING(build_qt);

        // iterate over the size of the resulting joinrel
        for(int i=2; i<=n_rels; i++){
            // give possibility to user to interrupt
            CHECK_FOR_INTERRUPTS();
            
            // calculate number of combinations of relations that make up 
            // a joinrel of size i
            params.n_sets = BINOM(params.binoms, n_rels, n_rels, i);
            params.n_joins_per_set = (1<<i) - 2;
            params.tot = params.n_sets * params.n_joins_per_set;

            uint64_t n_iters;
            uint64_t filter_threshold = gpuqo_dpsub_n_parallel * gpuqo_dpsub_filter_threshold;
            uint64_t csg_threshold = gpuqo_dpsub_n_parallel * gpuqo_dpsub_csg_threshold;
            if (gpuqo_dpsub_csg_enable && params.tot > csg_threshold){
#if defined(GPUQO_DEBUG) || defined(GPUQO_PROFILE)
                printf("\nStarting filtered-csg iteration %d: %llu combinations\n", i, params.tot);
#endif
                n_iters = dpsub_filtered_iteration<dpsubEnumerateCsg>(i, params);
            } else if (gpuqo_dpsub_filter_enable && params.tot > filter_threshold){
#if defined(GPUQO_DEBUG) || defined(GPUQO_PROFILE)
                printf("\nStarting filtered iteration %d: %llu combinations\n", i, params.tot);
#endif
                n_iters = dpsub_filtered_iteration<dpsubEnumerateAllSubs>(i, params);
            } else {
#if defined(GPUQO_DEBUG) || defined(GPUQO_PROFILE)
                printf("\nStarting unfiltered iteration %d: %llu combinations\n", i, params.tot);
#endif
                n_iters = dpsub_unfiltered_iteration<dpsubEnumerateAllSubs>(i, params);
            }

#ifdef GPUQO_DEBUG
            printf("It took %d iterations\n", n_iters);
#endif
            PRINT_CHECKPOINT_TIMING(unrank);
            PRINT_CHECKPOINT_TIMING(filter);
            PRINT_CHECKPOINT_TIMING(compute);
            PRINT_CHECKPOINT_TIMING(prune);
            PRINT_CHECKPOINT_TIMING(scatter);
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
