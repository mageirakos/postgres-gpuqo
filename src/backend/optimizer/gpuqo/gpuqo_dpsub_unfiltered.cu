/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_unfiltered.cu
 *      declarations necessary for dpsub_unfiltered_iteration
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

/* unrankEvaluateDPSub
 *
 *	 unrank algorithm for DPsub GPU variant with embedded evaluation and 
 *   partial pruning.
 */
template<typename BinaryFunction>
struct unrankEvaluateDPSub : public thrust::unary_function< uint64_t,thrust::tuple<RelationID, JoinRelation> >
{
    thrust::device_ptr<uint64_t> binoms;
    int sq;
    int qss;
    uint64_t offset;
    int n_splits;
    BinaryFunction enum_functor;
public:
    unrankEvaluateDPSub(
        BinaryFunction _enum_functor,
        int _sq,
        thrust::device_ptr<uint64_t> _binoms,
        int _qss,
        uint64_t _offset,
        int _n_splits
    ) : enum_functor(_enum_functor), sq(_sq), binoms(_binoms), 
        qss(_qss), offset(_offset), n_splits(_n_splits)
    {}
 
    __device__
    thrust::tuple<RelationID, JoinRelation> operator()(uint64_t tid)
    {
        uint64_t real_id = tid + offset;
        uint64_t sid = real_id / n_splits;
        uint64_t cid = real_id % n_splits;

        LOG_DEBUG("[%llu] n_splits=%d, sid=%llu, cid=[%llu,%llu)\n", 
                tid, n_splits, sid, cid, cid+ceil_div((1ULL)<<qss, n_splits));

        RelationID s = dpsub_unrank_sid(sid, qss, sq, binoms.get());
        RelationID relid = s<<1;

        LOG_DEBUG("[%llu] s=%llu\n", tid, s);

        JoinRelation jr_out = enum_functor(relid, cid);
        return thrust::tuple<RelationID, JoinRelation>(relid, jr_out);
    }
};

int dpsub_unfiltered_iteration(int iter, dpsub_iter_param_t &params){
    uint64_t n_joins_per_thread;
    uint64_t n_sets_per_iteration;
    uint64_t threads_per_set;
    uint64_t factor = gpuqo_dpsub_n_parallel / params.n_sets;

    if (factor < 32 || params.n_joins_per_set <= 32){
        threads_per_set = 32;
    } else{
        threads_per_set = BMS64_HIGHEST(min(factor, params.n_joins_per_set));
    }
    
    n_joins_per_thread = ceil_div(params.n_joins_per_set, threads_per_set);
    n_sets_per_iteration = min(gpuqo_dpsub_n_parallel / threads_per_set, params.n_sets);

    LOG_PROFILE("n_joins_per_thread=%llu, n_sets_per_iteration=%llu, threads_per_set=%llu, factor=%llu\n",
            n_joins_per_thread,
            n_sets_per_iteration,
            threads_per_set,
            factor
        );

    uint64_t id_offset = 0;
    uint64_t offset = 0;
    int n_iters = 0;
    while (offset < params.tot){
        uint64_t n_threads;
        uint64_t n_remaining_sets = (params.tot-offset)/params.n_joins_per_set;
        if (n_remaining_sets >= n_sets_per_iteration){
            n_threads = n_sets_per_iteration*threads_per_set;
        } else {
            n_threads = n_remaining_sets*threads_per_set;
        }   

        START_TIMING(unrank);
        // fill scratchpad
        thrust::tabulate(
            thrust::make_zip_iterator(thrust::make_tuple(
                params.gpu_scratchpad_keys.begin(),
                params.gpu_scratchpad_vals.begin()
            )),
            thrust::make_zip_iterator(thrust::make_tuple(
                params.gpu_scratchpad_keys.begin()+n_threads,
                params.gpu_scratchpad_vals.begin()+n_threads
            )),
            unrankEvaluateDPSub<dpsubEnumerateAllSubs>(
                dpsubEnumerateAllSubs(
                    params.gpu_memo_vals.data(),
                    params.info,
                    threads_per_set
                ),
                params.info->n_rels,
                params.gpu_binoms.data(),
                iter,
                id_offset,
                threads_per_set
            ) 
        );
        STOP_TIMING(unrank);

        LOG_DEBUG("After tabulate\n");
        DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_threads);
        DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_threads);

        dpsub_prune_scatter(threads_per_set, n_threads, params);

        n_iters++;
        id_offset += n_threads;
        offset += n_sets_per_iteration*params.n_joins_per_set;
    } // loop: while(offset<tot)

    return n_iters;
}
