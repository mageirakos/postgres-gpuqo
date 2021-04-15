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
template<typename BitmapsetN>
__global__
void unrankFilteredDPSubKernel(int sq, int qss, 
                               uint_t<BitmapsetN> offset, uint32_t n_tab_sets,
                               uint_t<BitmapsetN>* binoms,
                               GpuqoPlannerInfo<BitmapsetN>* info,
                               BitmapsetN* out_relids)
{
    uint32_t threadid = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t n_threads = blockDim.x * gridDim.x;

    int n_active = __popc(__activemask());
    __shared__ BitmapsetN edge_table[BitmapsetN::SIZE];
    for (int i = threadIdx.x; i < sq; i+=n_active){
        edge_table[i] = info->edge_table[i];
    }
    __syncthreads();
    
    if (threadid < n_tab_sets){
        uint_t<BitmapsetN> sets_per_thread = ceil_div(n_tab_sets, n_threads);
        uint_t<BitmapsetN> n_excess = n_tab_sets % n_threads;
        uint_t<BitmapsetN> idx;
        if (threadid < n_excess){
            idx = threadid * sets_per_thread + offset;
        } else {
            idx = n_excess * sets_per_thread 
                    + (threadid - n_excess) * (sets_per_thread-1) 
                    + offset;
        }
        
        
        BitmapsetN s = dpsub_unrank_sid<BitmapsetN>(idx, qss, sq, binoms);

        for (uint32_t tid = threadid; tid < n_tab_sets; tid += n_threads){
            BitmapsetN relid = s << 1;
            
            if (!is_connected(relid, edge_table))
                relid = BitmapsetN(0);
            
            LOG_DEBUG("[%u,%u] tid=%u idx=%u s=%u relid=%u\n", 
                        blockIdx.x, threadIdx.x, 
                        tid, idx++, s.toUint(), relid.toUint());
            out_relids[tid] = relid;

            s = dpsub_unrank_next(s);
        }
    }
}

template<typename BitmapsetN>
static void launchUnrankFilteredDPSubKernel(int sq, int qss, 
                                     uint64_t offset, 
                                     uint32_t n_tab_sets,
                                     uint_t<BitmapsetN>* binoms,
                                     GpuqoPlannerInfo<BitmapsetN>* info,
                                     BitmapsetN* out_relids)
{
    int blocksize = 512;
    
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, 
        unrankFilteredDPSubKernel<BitmapsetN>, 0, blocksize);

    int gridsize = min(mingridsize, ceil_div(n_tab_sets, blocksize));

    unrankFilteredDPSubKernel<<<gridsize, blocksize>>>(
        sq, qss, 
        offset, n_tab_sets,
        binoms,
        info,
        out_relids
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err){
        printf("CUDA ERROR! %s: %s\n", 
            cudaGetErrorName(err),
            cudaGetErrorString(err)
        );
        throw "CUDA ERROR";
    }
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
        LOG_DEBUG("red<%3d> idx[%3d]=%3d, cost[%3d]=%.2f\n",
                    STEP, tid, s_indexes[tid], tid, s_costs[tid]);
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
template<typename BitmapsetN, int n_splits, typename Function, bool full_shmem>
__global__
void evaluateFilteredDPSubKernel(BitmapsetN* pending_keys, BitmapsetN* scratchpad_keys, JoinRelation<BitmapsetN>* scratchpad_vals, uint32_t sq, uint32_t qss, uint32_t n_pending_sets, uint32_t n_sets, HashTableDpsub<BitmapsetN> memo, GpuqoPlannerInfo<BitmapsetN> *info){
    uint32_t n_threads_cuda = blockDim.x * gridDim.x;

    __shared__ volatile float shared_costs[BLOCK_DIM];
    __shared__ volatile int shared_idxs[BLOCK_DIM];

    extern __shared__ uint64_t dynshmem[];
    int size = full_shmem ? info->size : sizeof(GpuqoPlannerInfo<BitmapsetN>);
    for (int i = threadIdx.x; i < size/8; i+=blockDim.x){
        dynshmem[i] = *((uint64_t*)info + i);
    }
    __syncthreads();
    
    GpuqoPlannerInfo<BitmapsetN> *info_sh = (GpuqoPlannerInfo<BitmapsetN> *)dynshmem;
	
    if (threadIdx.x == 0){
        char *p = (char*) dynshmem;
        p += sizeof(GpuqoPlannerInfo<BitmapsetN>);

        if (full_shmem){
            info_sh->fk_selec_idxs = (unsigned int*)p;
            p += sizeof(unsigned int) * info->n_fk_selecs;
            
            info_sh->fk_selec_sels = (float*) p;
            p += sizeof(float) * info->n_fk_selecs;
            
            info_sh->eq_classes = (BitmapsetN*) p;
            p += sizeof(BitmapsetN) * info->n_eq_classes;
            
            info_sh->eq_class_sels = (float*) p;
            p += sizeof(float) * info->n_eq_class_sels;
        } else {
            info_sh->fk_selec_idxs = info->fk_selec_idxs;
            info_sh->fk_selec_sels = info->fk_selec_sels;
            info_sh->eq_classes = info->eq_classes;
            info_sh->eq_class_sels = info->eq_class_sels;
        }
    }
    __syncthreads();
    
    for (uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x; 
        tid < n_splits*n_sets; 
        tid += n_threads_cuda) 
    {
        uint32_t rid = n_pending_sets - 1 - (tid / n_splits);
        uint32_t cid = tid % n_splits;

        Assert(n_pending_sets-1 <= 0xFFFFFFFF - tid / n_splits);

        BitmapsetN relid = pending_keys[rid];
        Assert(is_connected(relid, info->edge_table));

        LOG_DEBUG("[%u] n_splits=%d, rid=%u, cid=%u, relid=%u\n", 
                tid, n_splits, rid, cid, relid.toUint());
        
        JoinRelation<BitmapsetN> jr_out = Function{}(relid, cid, n_splits, 
                                        memo, info_sh);
        shared_idxs[threadIdx.x] = threadIdx.x;
        shared_costs[threadIdx.x] = jr_out.cost;

        if (n_splits > WARP_SIZE)
            __syncthreads();
        else
            __syncwarp();

        LOG_DEBUG("red<%3d> before: idx[%3d]=%3d, cost[%3d]=%.2f\n",
                n_splits, threadIdx.x, shared_idxs[threadIdx.x], 
                        threadIdx.x, shared_costs[threadIdx.x]);

        blockReduceMin<n_splits>(&shared_idxs[0], &shared_costs[0]);

        int leader = threadIdx.x & (~(n_splits-1));

        if (threadIdx.x == leader){
            LOG_DEBUG("red<%3d> after: idx[%3d]=%3d, cost[%3d]=%.2f\n",
                n_splits, threadIdx.x, shared_idxs[threadIdx.x], 
                        threadIdx.x, shared_costs[threadIdx.x]);
        }

        if (threadIdx.x == shared_idxs[leader]){
            LOG_DEBUG("[%3d] write scratch[%d] = %u (l=%u, r=%u, cost=%.2f)\n",
                threadIdx.x, tid/n_splits, relid.toUint(), 
                jr_out.left_rel_id.toUint(), jr_out.right_rel_id.toUint(), jr_out.cost);
            scratchpad_keys[tid/n_splits] = relid;
            scratchpad_vals[tid/n_splits] = jr_out;
        }
    }
}

template<typename BitmapsetN, int n_splits, typename Function, bool full_shmem>
static void __launchEvaluateFilteredDPSubKernel(BitmapsetN* pending_keys, BitmapsetN* scratchpad_keys, JoinRelation<BitmapsetN>* scratchpad_vals, uint32_t sq, uint32_t qss, uint32_t n_pending_sets, uint32_t n_sets, uint32_t shmem_size, HashTableDpsub<BitmapsetN> &memo, GpuqoPlannerInfo<BitmapsetN> *info)
{
    int blocksize = BLOCK_DIM;
    
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, 
        evaluateFilteredDPSubKernel<BitmapsetN, n_splits, Function, full_shmem>, 0, blocksize);

    int gridsize = min(mingridsize, ceil_div(n_sets*n_splits, blocksize));
        
    // cudaFuncSetCacheConfig(evaluateFilteredDPSubKernel<n_splits, Function>, cudaFuncCachePreferL1);

    // n_splits is a power of 2 and is lower than or equal to BLOCK_DIM
    Assert(popc(n_splits) == 1 && n_splits <= BLOCK_DIM);

    evaluateFilteredDPSubKernel<BitmapsetN, n_splits, Function, full_shmem><<<gridsize, blocksize, shmem_size>>>(
        pending_keys, scratchpad_keys, scratchpad_vals,
        sq, qss, 
        n_pending_sets, n_sets,
        memo, info
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err){
        printf("CUDA ERROR! %s: %s\n", 
            cudaGetErrorName(err),
            cudaGetErrorString(err)
        );
        throw "CUDA ERROR";
    }
}

template<typename BitmapsetN, int n_splits, typename Function>
static void _launchEvaluateFilteredDPSubKernel(BitmapsetN* pending_keys, BitmapsetN* scratchpad_keys, JoinRelation<BitmapsetN>* scratchpad_vals, uint32_t sq, uint32_t qss, uint32_t n_pending_sets, uint32_t n_sets, HashTableDpsub<BitmapsetN> &memo, GpuqoPlannerInfo<BitmapsetN> *info, GpuqoPlannerInfo<BitmapsetN> *gpu_info)
{
    if (info->size > 6000){ //TODO check with device capability
        int shmem_size = sizeof(GpuqoPlannerInfo<BitmapsetN>);
        __launchEvaluateFilteredDPSubKernel<BitmapsetN, n_splits, Function, false>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, shmem_size, memo, gpu_info);
    } else {
        int shmem_size = info->size;
        __launchEvaluateFilteredDPSubKernel<BitmapsetN, n_splits, Function, true>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, shmem_size, memo, gpu_info);

    }
}


template<typename BitmapsetN, typename Function>
static void launchEvaluateFilteredDPSubKernel(BitmapsetN* pending_keys, BitmapsetN* scratchpad_keys, JoinRelation<BitmapsetN>* scratchpad_vals, uint32_t sq, uint32_t qss, uint32_t n_pending_sets, uint32_t n_splits, uint32_t n_sets, HashTableDpsub<BitmapsetN> &memo, GpuqoPlannerInfo<BitmapsetN> *info, GpuqoPlannerInfo<BitmapsetN> *gpu_info){
    switch(n_splits){
    case    1:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,   1, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case    2:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,   2, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case    4:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,   4, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case    8:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,   8, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case   16:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,  16, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case   32:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,  32, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case   64:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,  64, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case  128:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN, 128, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case  256:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN, 256, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case  512:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN, 512, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    case 1024:
        _launchEvaluateFilteredDPSubKernel<BitmapsetN,1024, Function>(pending_keys, scratchpad_keys, scratchpad_vals, sq, qss, n_pending_sets, n_sets, memo, info, gpu_info);
        break;
    default:
        printf("FATAL ERROR: Trying to call launchEvaluateFilteredDPSubKernel with n_splits=%d\n", n_splits);
        exit(1);
    }
}

template<typename BitmapsetN>
static uint32_t dpsub_generic_graph_evaluation(int iter, 
                                    uint64_t n_remaining_sets,
                                    uint64_t offset, uint32_t n_pending_sets, 
                                    dpsub_iter_param_t<BitmapsetN> &params)
{
    uint64_t n_joins_per_thread;
    uint32_t n_sets_per_iteration;
    uint32_t threads_per_set;
    uint32_t factor = gpuqo_n_parallel / n_pending_sets;
    
    threads_per_set = floorPow2(min((uint64_t) factor, params.n_joins_per_set));
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
            launchEvaluateFilteredDPSubKernel<BitmapsetN,dpsubEnumerateCsg<BitmapsetN> >(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                *params.memo,
                params.info,
                params.gpu_info
            );
        } else {
            launchEvaluateFilteredDPSubKernel<BitmapsetN,dpsubEnumerateAllSubs<BitmapsetN> >(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                *params.memo,
                params.info,
                params.gpu_info
            );
        }           
        STOP_TIMING(compute);

        LOG_DEBUG("After tabulate\n");
        DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_eval_sets);
        DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_eval_sets);

        dpsub_scatter<BitmapsetN>(n_eval_sets, params);

        n_pending_sets -= n_eval_sets;
    }

    return n_pending_sets;
}

template<typename BitmapsetN>
static uint32_t dpsub_bicc_evaluation(int iter, uint64_t n_remaining_sets,
                                    uint64_t offset, uint32_t n_pending_sets, 
                                    dpsub_iter_param_t<BitmapsetN> &params)
{
    uint64_t n_joins_per_thread;
    uint32_t n_sets_per_iteration;
    uint32_t threads_per_set;
    uint32_t factor = gpuqo_n_parallel / n_pending_sets;

    threads_per_set = WARP_SIZE;
    
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
        launchEvaluateFilteredDPSubKernel<BitmapsetN,dpsubEnumerateBiCC<BitmapsetN> >(
            thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
            thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
            thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
            params.info->n_rels,
            iter,
            n_pending_sets,
            threads_per_set,
            n_eval_sets,
            *params.memo,
            params.info,
            params.gpu_info
        );     
        STOP_TIMING(compute);

        LOG_DEBUG("After tabulate\n");
        DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_eval_sets);
        DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_eval_sets);

        dpsub_scatter<BitmapsetN>(n_eval_sets, params);

        n_pending_sets -= n_eval_sets;
    }

    return n_pending_sets;
}

template<typename BitmapsetN>
static uint32_t dpsub_tree_evaluation(int iter, uint64_t n_remaining_sets, 
                           uint64_t offset, uint32_t n_pending_sets, 
                           dpsub_iter_param_t<BitmapsetN> &params)
{
    uint64_t n_joins_per_thread;
    uint32_t n_sets_per_iteration;
    uint32_t threads_per_set;
    uint32_t factor = gpuqo_n_parallel / n_pending_sets;
    uint32_t n_joins_per_set = iter; 

    threads_per_set = min(max(1, factor), n_joins_per_set);
    threads_per_set = min(threads_per_set, BLOCK_DIM); // at most block size
    threads_per_set = floorPow2(threads_per_set); // round to closest pow2
    
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
            launchEvaluateFilteredDPSubKernel<BitmapsetN,dpsubEnumerateTreeWithSubtrees<BitmapsetN> >(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                *params.memo,
                params.info,
                params.gpu_info
            );
        } else {
            launchEvaluateFilteredDPSubKernel<BitmapsetN,dpsubEnumerateTreeSimple<BitmapsetN> >(
                thrust::raw_pointer_cast(params.gpu_pending_keys.data())+offset,
                thrust::raw_pointer_cast(params.gpu_scratchpad_keys.data()),
                thrust::raw_pointer_cast(params.gpu_scratchpad_vals.data()),
                params.info->n_rels,
                iter,
                n_pending_sets,
                threads_per_set,
                n_eval_sets,
                *params.memo,
                params.info,
                params.gpu_info
            );
        }
                    
        STOP_TIMING(compute);

        LOG_DEBUG("After tabulate\n");
        DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_eval_sets);
        DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_eval_sets);

        dpsub_scatter<BitmapsetN>(n_eval_sets, params);

        n_pending_sets -= n_eval_sets;
    }

    return n_pending_sets;
}

template<typename BitmapsetN>
int dpsub_filtered_iteration(int iter, dpsub_iter_param_t<BitmapsetN> &params){   
    int n_iters = 0;
    uint64_t set_offset = 0;
    uint32_t n_pending_sets = 0;
    while (set_offset < params.n_sets){
        uint64_t n_remaining_sets = params.n_sets - set_offset;
        
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
                thrust::host_vector<BitmapsetN> relids(n_tab_sets);
                uint64_t n_valid_relids = 0;
                BitmapsetN s = dpsub_unrank_sid<BitmapsetN>(0, iter, params.info->n_rels, params.binoms.data());
                for (uint32_t sid=0; sid < n_tab_sets; sid++){
                    BitmapsetN relid = s << 1;
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
                    params.gpu_info,
                    thrust::raw_pointer_cast(params.gpu_pending_keys.data())+n_pending_sets

                );
                STOP_TIMING(unrank);

                START_TIMING(filter);
                auto keys_end_iter = thrust::remove(
                    params.gpu_pending_keys.begin()+n_pending_sets,
                    params.gpu_pending_keys.begin()+(n_pending_sets+n_tab_sets),
                    BitmapsetN(0)
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
                findCycleInRelation<BitmapsetN>(params.gpu_info)
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

template int dpsub_filtered_iteration<Bitmapset32>(int iter, dpsub_iter_param_t<Bitmapset32> &params);
template int dpsub_filtered_iteration<Bitmapset64>(int iter, dpsub_iter_param_t<Bitmapset64> &params);