/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_filtered.cu
 *      declarations necessary for dpsub_filtered_iteration
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_filtered.cu
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
#include "gpuqo_dpsub_enum_all_subs.cuh"
#include "gpuqo_dpsub_csg.cuh"
#include "gpuqo_dpsub_tree.cuh"
#include "gpuqo_dpsub_bicc.cuh"

// user-configured variables (generic)
bool gpuqo_dpsub_filter_enable;
int gpuqo_dpsub_filter_threshold;
int gpuqo_dpsub_filter_cpu_enum_threshold;
int gpuqo_dpsub_filter_keys_overprovisioning;

// user-configured variables (csg)
bool gpuqo_dpsub_csg_enable;
int gpuqo_dpsub_csg_threshold;

// user-configured variables (tree)
bool gpuqo_dpsub_tree_enable;

// user-configured variables (bicc)
bool gpuqo_dpsub_bicc_enable;

/**
    Faster dpsub enumeration using bit magic to compute next set.
 */
__global__
void unrankFilteredDPSubKernel(int sq, int qss, 
                               uint32_t offset, uint32_t n_tab_sets,
                               uint32_t* binoms,
                               EdgeMask* gobal_edge_table,
                               RelationID* out_relids)
{
    uint32_t threadid = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t n_threads = blockDim.x * gridDim.x;

    int n_active = __popc(__activemask());
    __shared__ EdgeMask edge_table[32];
    for (int i = threadIdx.x; i < sq; i+=n_active){
        edge_table[i] = gobal_edge_table[i];
    }
    __syncthreads();
    
    if (threadid < n_tab_sets){
        uint32_t sets_per_thread = ceil_div(n_tab_sets, n_threads);
        uint32_t n_excess = n_tab_sets % n_threads;
        uint32_t idx;
        if (threadid < n_excess){
            idx = threadid * sets_per_thread + offset;
        } else {
            idx = n_excess * sets_per_thread 
                    + (threadid - n_excess) * (sets_per_thread-1) 
                    + offset;
        }
        
        
        RelationID s = dpsub_unrank_sid(idx, qss, sq, binoms);

        for (uint32_t tid = threadid; tid < n_tab_sets; tid += n_threads){
            RelationID relid = s << 1;
            
            if (!is_connected(relid, edge_table))
                relid = BMS32_EMPTY;
            
            LOG_DEBUG("[%u,%u] tid=%u idx=%u s=%u relid=%u\n", 
                        blockIdx.x, threadIdx.x, 
                        tid, idx++, s, relid);
            out_relids[tid] = relid;

            s = dpsub_unrank_next(s);
        }
    }
}

void launchUnrankFilteredDPSubKernel(int sq, int qss, 
                                     uint32_t offset, uint32_t n_tab_sets,
                                     uint32_t* binoms,
                                     EdgeMask* global_edge_table,
                                     RelationID* out_relids)
{
    int blocksize = 512;
    
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, 
        unrankFilteredDPSubKernel, 0, blocksize);

    int gridsize = min(mingridsize, ceil_div(n_tab_sets, blocksize));

    unrankFilteredDPSubKernel<<<gridsize, blocksize>>>(
        sq, qss, 
        offset, n_tab_sets,
        binoms,
        global_edge_table,
        out_relids
    );
}

template<int STEP>
__device__ void blockReduceMinStep(volatile int* s_indexes, 
                                volatile float* s_costs)
{
    int tid = threadIdx.x;

    if (tid % (2*STEP) == 0){
        if (s_costs[tid+STEP] < s_costs[tid]){
            s_costs[tid] = s_costs[tid+STEP];
            s_indexes[tid] = s_indexes[tid+STEP];
        }
    }
    if (STEP >= WARP_SIZE)
        __syncthreads();
    else
        __syncwarp();

}

template<int DIM>
__device__ void blockReduceMin(volatile int* s_indexes, 
                                volatile float* s_costs)
{
    if (DIM >   1) blockReduceMinStep<  1>(s_indexes, s_costs);
    if (DIM >   2) blockReduceMinStep<  2>(s_indexes, s_costs);
    if (DIM >   4) blockReduceMinStep<  4>(s_indexes, s_costs);
    if (DIM >   8) blockReduceMinStep<  8>(s_indexes, s_costs);
    if (DIM >  16) blockReduceMinStep< 16>(s_indexes, s_costs);
    if (DIM >  32) blockReduceMinStep< 32>(s_indexes, s_costs);
    if (DIM >  64) blockReduceMinStep< 64>(s_indexes, s_costs);
    if (DIM > 128) blockReduceMinStep<128>(s_indexes, s_costs);
    if (DIM > 256) blockReduceMinStep<256>(s_indexes, s_costs);
    if (DIM > 512) blockReduceMinStep<512>(s_indexes, s_costs);
}

 /* evaluateDPSub
  *
  *	 evaluation algorithm for DPsub GPU variant with partial pruning
  */
template<int n_splits, typename BinaryFunction>
__global__
void evaluateFilteredDPSubKernel(RelationID* pending_keys, RelationID* scratchpad_keys, JoinRelation* scratchpad_vals, int sq, int qss, uint32_t n_pending_sets, int n_sets, BinaryFunction enum_functor){
    uint32_t n_threads_cuda = blockDim.x * gridDim.x;

    __shared__ volatile float shared_costs[BLOCK_DIM];
    __shared__ volatile int shared_idxs[BLOCK_DIM];
    
    for (uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x; 
        tid < n_splits*n_sets; 
        tid += n_threads_cuda) 
    {
        uint32_t rid = n_pending_sets - 1 - (tid / n_splits);
        uint32_t cid = tid % n_splits;

        Assert(n_pending_sets-1 <= 0xFFFFFFFF - tid / n_splits);

        RelationID relid = pending_keys[rid];

        LOG_DEBUG("[%u] n_splits=%d, rid=%u, cid=%u, relid=%u\n", 
                tid, n_splits, rid, cid, relid);
        
        JoinRelation jr_out = enum_functor(relid, cid);
        shared_idxs[threadIdx.x] = threadIdx.x;
        shared_costs[threadIdx.x] = jr_out.cost;

        if (n_splits > WARP_SIZE)
            __syncthreads();
        else
            __syncwarp();

        blockReduceMin<n_splits>(&shared_idxs[0], &shared_costs[0]);

        int leader = threadIdx.x & (~(n_splits-1));

        if (threadIdx.x == shared_idxs[leader]){
            scratchpad_keys[tid/n_splits] = relid;
            scratchpad_vals[tid/n_splits] = jr_out;
        }
    }
}

template<int n_splits, typename BinaryFunction>
void _launchEvaluateFilteredDPSubKernel(RelationID* pending_keys, RelationID* scratchpad_keys, JoinRelation* scratchpad_vals, int sq, int qss, uint32_t n_pending_sets, int n_sets, BinaryFunction enum_functor)
{
    int blocksize = BLOCK_DIM;
    
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, 
        evaluateFilteredDPSubKernel<n_splits, BinaryFunction>, 0, blocksize);

    int gridsize = min(mingridsize, ceil_div(n_sets*n_splits, blocksize));
        
    cudaFuncSetCacheConfig(evaluateFilteredDPSubKernel<n_splits, BinaryFunction>, cudaFuncCachePreferL1);

    // n_splits is a power of 2 and is lower than or equal to BLOCK_DIM
    Assert(BMS32_SIZE(n_splits) == 1 && n_splits <= BLOCK_DIM);

    evaluateFilteredDPSubKernel<n_splits, BinaryFunction><<<gridsize, blocksize>>>(
        pending_keys, scratchpad_keys, scratchpad_vals,
        sq, qss, 
        n_pending_sets, n_sets,
        enum_functor
    );
}

template<typename BinaryFunction>
void launchEvaluateFilteredDPSubKernel(RelationID* pending_keys, RelationID* scratchpad_keys, JoinRelation* scratchpad_vals, int sq, int qss, uint32_t n_pending_sets, int n_splits, int n_sets, BinaryFunction enum_functor){
    switch(n_splits){
    case    1:
        _launchEvaluateFilteredDPSubKernel<   1, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case    2:
        _launchEvaluateFilteredDPSubKernel<   2, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case    4:
        _launchEvaluateFilteredDPSubKernel<   4, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case    8:
        _launchEvaluateFilteredDPSubKernel<   8, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case   16:
        _launchEvaluateFilteredDPSubKernel<  16, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case   32:
        _launchEvaluateFilteredDPSubKernel<  32, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case   64:
        _launchEvaluateFilteredDPSubKernel<  64, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case  128:
        _launchEvaluateFilteredDPSubKernel< 128, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case  256:
        _launchEvaluateFilteredDPSubKernel< 256, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case  512:
        _launchEvaluateFilteredDPSubKernel< 512, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    case 1024:
        _launchEvaluateFilteredDPSubKernel<1024, BinaryFunction>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, enum_functor);
        break;
    }
}


uint32_t dpsub_generic_graph_evaluation(int iter, uint32_t n_remaining_sets,
                                    uint32_t offset, uint32_t n_pending_sets, 
                                    dpsub_iter_param_t &params)
{
    uint32_t n_joins_per_thread;
    uint32_t n_sets_per_iteration;
    uint32_t threads_per_set;
    uint32_t factor = gpuqo_n_parallel / n_pending_sets;
    
    threads_per_set = BMS32_HIGHEST(min(factor, params.n_joins_per_set));
    threads_per_set = min(threads_per_set, BLOCK_DIM); // at most block size
    threads_per_set = max(threads_per_set, WARP_SIZE); // at least warp size
    
    n_joins_per_thread = ceil_div(params.n_joins_per_set, threads_per_set);
    n_sets_per_iteration = min(params.scratchpad_size, n_pending_sets);

    LOG_PROFILE("n_joins_per_thread=%u, n_sets_per_iteration=%u, threads_per_set=%u, factor=%u\n",
        n_joins_per_thread,
        n_sets_per_iteration,
        threads_per_set,
        factor
    );

    bool use_csg = (gpuqo_dpsub_csg_enable && n_joins_per_thread >= gpuqo_dpsub_csg_threshold);

    if (use_csg){
        LOG_PROFILE("Using CSG enumeration\n");
    } else{
        LOG_PROFILE("Using all subsets enumeration\n");
    }

    // do not empty all pending sets if there are some sets still to 
    // evaluate, since I will do them in the next iteration
    // If no sets remain, then I will empty all pending
    while (n_pending_sets >= gpuqo_n_parallel 
        || (n_pending_sets > 0 && n_remaining_sets == 0)
    ){
        uint32_t n_eval_sets = min(n_sets_per_iteration, n_pending_sets);

        START_TIMING(compute);
        if (use_csg) {
            launchEvaluateFilteredDPSubKernel(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                dpsubEnumerateCsg(
                    *params.memo,
                    params.info,
                    threads_per_set
                )
            );
        } else {
            launchEvaluateFilteredDPSubKernel(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                dpsubEnumerateAllSubs(
                    *params.memo,
                    params.info,
                    threads_per_set
                )  
            );
        }           
        STOP_TIMING(compute);

        LOG_DEBUG("After tabulate\n");
        DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_threads);
        DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_threads);

        dpsub_scatter(n_eval_sets, params);

        n_pending_sets -= n_eval_sets;
    }

    return n_pending_sets;
}


uint32_t dpsub_bicc_evaluation(int iter, uint32_t n_remaining_sets,
                                    uint32_t offset, uint32_t n_pending_sets, 
                                    dpsub_iter_param_t &params)
{
    uint32_t n_joins_per_thread;
    uint32_t n_sets_per_iteration;
    uint32_t threads_per_set;
    uint32_t factor = gpuqo_n_parallel / n_pending_sets;

    threads_per_set = 32;
    
    n_joins_per_thread = ceil_div(params.n_joins_per_set, threads_per_set);
    n_sets_per_iteration = min(params.scratchpad_size, n_pending_sets);

    LOG_PROFILE("n_joins_per_thread=%u, n_sets_per_iteration=%u, threads_per_set=%u, factor=%u\n",
        n_joins_per_thread,
        n_sets_per_iteration,
        threads_per_set,
        factor
    );
    LOG_PROFILE("Using BiCC enumeration\n");

    // do not empty all pending sets if there are some sets still to 
    // evaluate, since I will do them in the next iteration
    // If no sets remain, then I will empty all pending
    while (n_pending_sets >= gpuqo_n_parallel 
        || (n_pending_sets > 0 && n_remaining_sets == 0)
    ){
        uint32_t n_eval_sets = min(n_sets_per_iteration, n_pending_sets);

        START_TIMING(compute);
        launchEvaluateFilteredDPSubKernel(
            thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
            thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
            thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
            params.info->n_rels,
            iter,
            n_pending_sets,
            threads_per_set,
            n_eval_sets,
            dpsubEnumerateBiCC(
                *params.memo,
                params.info,
                threads_per_set
            )
        );     
        STOP_TIMING(compute);

        LOG_DEBUG("After tabulate\n");
        DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_threads);
        DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_threads);

        dpsub_scatter(n_eval_sets, params);

        n_pending_sets -= n_eval_sets;
    }

    return n_pending_sets;
}


uint32_t dpsub_tree_evaluation(int iter, uint32_t n_remaining_sets, 
                           uint32_t offset, uint32_t n_pending_sets, 
                           dpsub_iter_param_t &params)
{
    uint32_t n_joins_per_thread;
    uint32_t n_sets_per_iteration;
    uint32_t threads_per_set;
    uint32_t factor = gpuqo_n_parallel / n_pending_sets;
    uint32_t n_joins_per_set = iter; 

    threads_per_set = min(max(1, factor), n_joins_per_set);
    threads_per_set = min(threads_per_set, BLOCK_DIM); // at most block size
    
    n_joins_per_thread = ceil_div(n_joins_per_set, threads_per_set);
    n_sets_per_iteration = min(params.scratchpad_size, n_pending_sets);

    LOG_PROFILE("n_joins_per_thread=%u, n_sets_per_iteration=%u, threads_per_set=%u, factor=%u\n",
        n_joins_per_thread,
        n_sets_per_iteration,
        threads_per_set,
        factor
    );

    LOG_PROFILE("Using tree enumeration\n");

    // do not empty all pending sets if there are some sets still to 
    // evaluate, since I will do them in the next iteration
    // If no sets remain, then I will empty all pending
    while (n_pending_sets >= gpuqo_n_parallel 
        || (n_pending_sets > 0 && n_remaining_sets == 0)
    ){
        uint32_t n_eval_sets = min(n_sets_per_iteration, n_pending_sets);

        START_TIMING(compute);
        if (gpuqo_spanning_tree_enable){
            launchEvaluateFilteredDPSubKernel(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                dpsubEnumerateTreeWithSubtrees(                
                    *params.memo,
                    params.info,
                    threads_per_set
                ) 
            );
        } else {
            launchEvaluateFilteredDPSubKernel(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                dpsubEnumerateTreeSimple(
                    *params.memo,
                    params.info,
                    threads_per_set
                )
            );
        }
                    
        STOP_TIMING(compute);

        LOG_DEBUG("After tabulate\n");
        DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_threads);
        DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_threads);

        dpsub_scatter(n_eval_sets, params);

        n_pending_sets -= n_eval_sets;
    }

    return n_pending_sets;
}


int dpsub_filtered_iteration(int iter, dpsub_iter_param_t &params){   
    int n_iters = 0;
    uint32_t set_offset = 0;
    uint32_t n_pending_sets = 0;
    while (set_offset < params.n_sets){
        uint32_t n_remaining_sets = params.n_sets - set_offset;
        
        while(n_pending_sets < params.scratchpad_size
                && n_remaining_sets > 0){
            uint32_t n_tab_sets;

            if (n_remaining_sets > PENDING_KEYS_SIZE(params)-n_pending_sets){
                n_tab_sets = PENDING_KEYS_SIZE(params)-n_pending_sets;
            } else {
                n_tab_sets = n_remaining_sets;
            }

            if (n_tab_sets == 1){
                // if it's only one it's the last one so it's valid
                params.gpu_pending_keys[n_pending_sets] = params.out_relid;
                n_pending_sets += 1;
            } else if (n_tab_sets <= gpuqo_dpsub_filter_cpu_enum_threshold) {
                // fill (valid) pending keys on CPU
                // if they are too few do not bother going to GPU

                START_TIMING(unrank);
                thrust::host_vector<RelationID> relids(n_tab_sets);
                uint32_t n_valid_relids = 0;
                RelationID s = dpsub_unrank_sid(0, iter, params.info->n_rels, params.binoms.data());
                for (uint32_t sid=0; sid < n_tab_sets; sid++){
                    RelationID relid = s << 1;
                    if (is_connected(relid, params.info->edge_table)){
                        relids[n_valid_relids++] = relid; 
                    }
                    s = dpsub_unrank_next(s);
                }
                thrust::copy(relids.begin(), relids.begin()+n_valid_relids, params.gpu_pending_keys.begin()+n_pending_sets);

                n_pending_sets += n_valid_relids;
                STOP_TIMING(unrank);
            } else {
                // fill pending keys and filter on GPU 
                START_TIMING(unrank);
                LOG_DEBUG("Unranking %u sets from offset %u\n", 
                            n_tab_sets, set_offset);
                launchUnrankFilteredDPSubKernel(
                    params.info->n_rels, iter,
                    set_offset, n_tab_sets,
                    thrust::raw_pointer_cast(params.gpu_binoms.data()),
                    params.info->edge_table,
                    thrust::raw_pointer_cast(params.gpu_pending_keys.data())+n_pending_sets

                );
                STOP_TIMING(unrank);

                START_TIMING(filter);
                auto keys_end_iter = thrust::remove(
                    params.gpu_pending_keys.begin()+n_pending_sets,
                    params.gpu_pending_keys.begin()+(n_pending_sets+n_tab_sets),
                    BMS32_EMPTY
                );
                STOP_TIMING(filter);

                n_pending_sets = thrust::distance(
                    params.gpu_pending_keys.begin(),
                    keys_end_iter
                );
            } 

            set_offset += n_tab_sets;
            n_remaining_sets -= n_tab_sets;
        }  
        
        if (gpuqo_dpsub_tree_enable){
            auto middle = params.gpu_pending_keys.begin();

            if (!gpuqo_spanning_tree_enable){
                // if I'm not forcing spanning trees, I need to partition the 
                // subsets in cycles and treed
                middle = thrust::partition(
                params.gpu_pending_keys.begin(),
                params.gpu_pending_keys.begin()+n_pending_sets,
                findCycleInRelation(params.info)
            );
            } // otherwise "middle" is just the beginning (all trees)

            int n_cyclic = thrust::distance(
                params.gpu_pending_keys.begin(),
                middle
            );

            LOG_PROFILE("Cyclic: %d, Trees: %d, Tot: %d\n", 
                n_cyclic, 
                n_pending_sets - n_cyclic, 
                n_pending_sets
            );

            uint32_t graph_pending = 0;
            uint32_t tree_pending = 0;

            // TODO: maybe I can run both kernels in parallel if I have few
            //       relations
            if (n_cyclic > 0){
                graph_pending = dpsub_generic_graph_evaluation(
                                    iter, n_remaining_sets, 
                                               0, n_cyclic, params);
            }

            if (n_pending_sets - n_cyclic > 0){
                tree_pending = dpsub_tree_evaluation(iter, n_remaining_sets,
                                      n_cyclic, n_pending_sets-n_cyclic, 
                                      params);
            }

            // recompact
            if (n_cyclic > 0 && tree_pending != 0){
                thrust::copy(middle, middle + tree_pending, 
                            params.gpu_pending_keys.begin() + graph_pending
                );
            }

            n_pending_sets = graph_pending + tree_pending;


        } else if (gpuqo_dpsub_bicc_enable){
            n_pending_sets = dpsub_bicc_evaluation(
                                        iter, n_remaining_sets, 
                                           0, n_pending_sets, params);
        } else {
            n_pending_sets = dpsub_generic_graph_evaluation(
                                        iter, n_remaining_sets, 
                                           0, n_pending_sets, params);
        }
        
        n_iters++;
    }

    return n_iters;
}
