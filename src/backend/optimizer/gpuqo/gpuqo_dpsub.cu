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
#include "optimizer/gpuqo_query_tree.cuh"

// relsize depends on algorithm
#define RELSIZE (sizeof(JoinRelation))

// User-configured option
int gpuqo_dpsub_n_parallel;

/* unrankEvaluateDPSub
 *
 *	 unrank algorithm for DPsub GPU variant with embedded evaluation and 
 *   partial pruning.
 */
struct unrankEvaluateDPSub : public thrust::unary_function< uint64_t,thrust::tuple<RelationID, JoinRelation> >
{
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    thrust::device_ptr<EdgeInfo> edge_table;
    thrust::device_ptr<uint64_t> binoms;
    int sq;
    int qss;
    uint64_t offset;
    int n_pairs;
public:
    unrankEvaluateDPSub(
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        int _sq,
        thrust::device_ptr<EdgeInfo> _edge_table,
        thrust::device_ptr<uint64_t> _binoms,
        int _qss,
        uint64_t _offset,
        int _n_pairs
    ) : memo_vals(_memo_vals), base_rels(_base_rels), sq(_sq), 
        edge_table(_edge_table), binoms(_binoms), qss(_qss), offset(_offset),
        n_pairs(_n_pairs)
    {}

    __device__
    thrust::tuple<RelationID, JoinRelation> operator()(uint64_t tid) 
    {
        uint64_t splits_per_qs = ceil_div((1<<qss) - 2, n_pairs);
        uint64_t real_id = tid + offset;
        uint64_t sid = real_id / splits_per_qs;
        uint64_t cid = (real_id % splits_per_qs)*n_pairs+1;

#ifdef GPUQO_DEBUG 
        printf("[%llu] splits_per_qs=%llu, sid=%llu, cid=[%llu,%llu)\n", tid, splits_per_qs, sid, cid, cid+n_pairs);
#endif

        RelationID s = BMS64_EMPTY;
        int t = 0;
        int qss_tmp = qss, sq_tmp = sq;

        while (sq_tmp > 0 && qss_tmp > 0){
            uint64_t o = binoms[(sq_tmp-1) * (sq+1) + (qss_tmp-1)];
            if (sid < o){
                s = BMS64_UNION(s, BMS64_NTH(t));
                qss_tmp--;
            } else {
                sid -= o;
            }
            t++;
            sq_tmp--;
        }

#ifdef GPUQO_DEBUG 
        printf("[%llu] s=%llu\n", tid, s);
#endif
        
        JoinRelation jr_out;
        jr_out.id = BMS64_EMPTY;
        jr_out.cost = INFD;
        RelationID relid = s<<1;
        RelationID l = BMS64_EXPAND_TO_MASK(cid, relid);
        RelationID r;
        for (int i = 0; i < n_pairs; i++){
            r = BMS64_DIFFERENCE(relid, l);
            
            if (l == BMS64_EMPTY || r == BMS64_EMPTY)
                break;

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
                
                if (are_connected(left_rel, right_rel, base_rels.get(), sq, edge_table.get())){

#ifdef GPUQO_DEBUG 
                printf("[%llu] Joining %llu and %llu\n", tid, l, r);
#endif

                    jr.rows = estimate_join_rows(jr, left_rel, right_rel,
                                        base_rels.get(), sq, edge_table.get());

                    jr.cost = compute_join_cost(jr, left_rel, right_rel,
                                        base_rels.get(), sq, edge_table.get());

                    if (jr.cost < jr_out.cost){
                        jr_out = jr;
                    }
                } else {
#ifdef GPUQO_DEBUG 
                    printf("[%llu] Cannot join %llu and %llu\n", tid, l, r);
#endif
                }
            } else {
#ifdef GPUQO_DEBUG 
                printf("[%llu] Invalid subsets %llu and %llu\n", tid, l, r);
                continue;
#endif
            }

            l = BMS64_NEXT_SUBSET(l, relid);
        }

        return thrust::tuple<RelationID, JoinRelation>(relid, jr_out);
    }
};




void precompute_binoms(thrust::host_vector<uint64_t> &binoms, int N){
    for (int n = 0; n <= N; n++){
        for (int k = 0; k <= N; k++){
            int idx = n*(N+1)+k;
            if (k > n){
                // invalid
                binoms[idx] = 0;
            } else if (k == 0 || k == n){
                // base case
                binoms[idx] = 1;
            } else {
                binoms[idx] = binoms[(n-1)*(N+1)+(k-1)] + binoms[(n-1)*(N+1)+k];
            }
        }

    }
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
    
    thrust::device_vector<BaseRelation> gpu_base_rels(base_rels, base_rels + n_rels);
    thrust::device_vector<EdgeInfo> gpu_edge_table(edge_table, edge_table + n_rels*n_rels);
    thrust::device_vector<JoinRelation> gpu_memo_vals(memo_size);
    QueryTree* out = NULL;
    RelationID out_relid = BMS64_EMPTY;

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
        gpu_memo_vals[base_rels[i].id] = t;

        out_relid = BMS64_UNION(out_relid, base_rels[i].id);
    }

    int binoms_size = (n_rels+1)*(n_rels+1);
    thrust::host_vector<uint64_t> binoms(binoms_size);
    precompute_binoms(binoms, n_rels);
    thrust::device_vector<uint64_t> gpu_binoms = binoms;

    // scratchpad size is increased on demand, starting from a minimum capacity
    uninit_device_vector_relid gpu_scratchpad_keys(gpuqo_dpsub_n_parallel);
    uninit_device_vector_joinrel gpu_scratchpad_vals(gpuqo_dpsub_n_parallel);
    uninit_device_vector_relid gpu_reduced_keys(gpuqo_dpsub_n_parallel);
    uninit_device_vector_joinrel gpu_reduced_vals(gpuqo_dpsub_n_parallel);

    STOP_TIMING(init);

#ifdef GPUQO_DEBUG
    printVector(gpu_binoms.begin(), gpu_binoms.end());    
#endif

    START_TIMING(execute);
    try{ // catch any exception in thrust
        DECLARE_TIMING(iter_init);
        DECLARE_NV_TIMING(unrank_compute);
        DECLARE_NV_TIMING(prune);
        DECLARE_NV_TIMING(scatter);
        DECLARE_NV_TIMING(build_qt);

        // iterate over the size of the resulting joinrel
        for(int i=2; i<=n_rels; i++){
            // give possibility to user to interrupt
            CHECK_FOR_INTERRUPTS();

            START_TIMING(iter_init);
            
            // calculate number of combinations of relations that make up 
            // a joinrel of size i
            uint64_t n_sets = binoms[n_rels * (n_rels+1) + i];
            uint64_t n_joins_per_set = (1<<i) - 2;
            uint64_t tot = n_sets * n_joins_per_set;
            
            uint64_t n_joins_per_thread;
            uint64_t n_sets_per_iteration;
            uint64_t factor = gpuqo_dpsub_n_parallel / n_sets;
            if (factor < 1){ // n_sets > gpuqo_dpsub_n_parallel
                n_joins_per_thread = n_joins_per_set;
                n_sets_per_iteration = gpuqo_dpsub_n_parallel;
            } else{
                n_sets_per_iteration = n_sets;
                n_joins_per_thread = ceil_div(n_joins_per_set, factor);
            }
            
            STOP_TIMING(iter_init);

#if defined(GPUQO_DEBUG) || defined(GPUQO_PROFILE)
            printf("\nStarting iteration %d: %llu combinations\n", i, tot);
#endif
            uint64_t id_offset = 0;
            uint64_t offset = 0;
            int n_iters = 0;
            while (offset < tot){
                uint64_t n_threads;
                uint64_t threads_per_set = ceil_div(n_joins_per_set, n_joins_per_thread);
                uint64_t n_remaining_sets = (tot-offset)/n_joins_per_set;
                if (n_remaining_sets >= n_sets_per_iteration){
                    n_threads = n_sets_per_iteration*threads_per_set;
                } else {
                    n_threads = n_remaining_sets*threads_per_set;
                }   

                START_TIMING(unrank_compute);
                // fill scratchpad
                thrust::tabulate(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        gpu_scratchpad_keys.begin(),
                        gpu_scratchpad_vals.begin()
                    )),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        gpu_scratchpad_keys.begin()+n_threads,
                        gpu_scratchpad_vals.begin()+n_threads
                    )),
                    unrankEvaluateDPSub(
                        gpu_memo_vals.data(),
                        gpu_base_rels.data(),
                        n_rels,
                        gpu_edge_table.data(),
                        gpu_binoms.data(),
                        i,
                        id_offset,
                        n_joins_per_thread
                    ) 
                );
                STOP_TIMING(unrank_compute);

#ifdef GPUQO_DEBUG
                printf("After tabulate\n");
                printVector(gpu_scratchpad_keys.begin(), gpu_scratchpad_keys.begin()+n_threads);
                printVector(gpu_scratchpad_vals.begin(), gpu_scratchpad_vals.begin()+n_threads);
#endif

                // give possibility to user to interrupt
                CHECK_FOR_INTERRUPTS();

                thrust::pair<uninit_device_vector_relid::iterator, uninit_device_vector_joinrel::iterator> scatter_from_iters;
                thrust::pair<uninit_device_vector_relid::iterator, uninit_device_vector_joinrel::iterator> scatter_to_iters;

                if (n_joins_per_thread < n_joins_per_set){
                    START_TIMING(prune);
                    scatter_from_iters = thrust::make_pair(
                        gpu_reduced_keys.begin(),
                        gpu_reduced_vals.begin()
                    );
                    // prune to intermediate memory
                    scatter_to_iters = thrust::reduce_by_key(
                        gpu_scratchpad_keys.begin(),
                        gpu_scratchpad_keys.begin() + n_threads,
                        gpu_scratchpad_vals.begin(),
                        gpu_reduced_keys.begin(),
                        gpu_reduced_vals.begin(),
                        thrust::equal_to<uint64_t>(),
                        thrust::minimum<JoinRelation>()
                    );
                    STOP_TIMING(prune);
                } else{
                    scatter_from_iters = thrust::make_pair(
                        gpu_scratchpad_keys.begin(),
                        gpu_scratchpad_vals.begin()
                    );
                    scatter_to_iters = thrust::make_pair(
                        (gpu_scratchpad_keys.begin()+n_threads),
                        (gpu_scratchpad_vals.begin()+n_threads)
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
                    gpu_memo_vals.begin()
                );
                STOP_TIMING(scatter);

                n_iters++;
                id_offset += n_threads;
                offset += n_sets_per_iteration*n_joins_per_set;
            } // loop: while(offset<tot)

#ifdef GPUQO_DEBUG
            printf("It took %d iterations\n", n_iters);
#endif
            
            PRINT_CHECKPOINT_TIMING(iter_init);
            PRINT_CHECKPOINT_TIMING(unrank_compute);
            PRINT_CHECKPOINT_TIMING(prune);
            PRINT_CHECKPOINT_TIMING(scatter);
        } // dpsub loop: for i = 2..n_rels

        START_TIMING(build_qt);
            
        buildQueryTree(out_relid, gpu_memo_vals, &out);
    
        STOP_TIMING(build_qt);
    
        PRINT_TOTAL_TIMING(iter_init);
        PRINT_TOTAL_TIMING(unrank_compute);
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
