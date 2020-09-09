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

// user-configured variables
bool gpuqo_dpsub_filter_enable;
int gpuqo_dpsub_filter_threshold;
int gpuqo_dpsub_filter_cpu_enum_threshold;
int gpuqo_dpsub_filter_keys_overprovisioning;

/* unrankDPSub
 *
 *	 unrank algorithm for DPsub GPU variant. 
 */
struct unrankFilteredDPSub : public thrust::unary_function< uint64_t, RelationID >
{
    thrust::device_ptr<uint64_t> binoms;
    int sq;
    int qss;
    uint64_t offset;
public:
    unrankFilteredDPSub(
        int _sq,
        thrust::device_ptr<uint64_t> _binoms,
        int _qss,
        uint64_t _offset
    ) : sq(_sq), binoms(_binoms), qss(_qss), offset(_offset)
    {}
 
    __device__
    RelationID operator()(uint64_t tid)
    {
        uint64_t sid = tid + offset;
        RelationID s = dpsub_unrank_sid(sid, qss, sq, binoms.get());
        
        LOG_DEBUG("[%llu] s=%llu\n", tid, s);
        
        RelationID relid = s<<1;
        return relid;
    }
 };
 
 /* evaluateDPSub
  *
  *	 evaluation algorithm for DPsub GPU variant with partial pruning
  */
template<typename BinaryFunction>
struct evaluateFilteredDPSub : public thrust::unary_function< uint64_t, thrust::tuple<RelationID, JoinRelation> >
{
    thrust::device_ptr<RelationID> pending_keys;
    int sq;
    int qss;
    uint64_t n_pending_sets;
    int n_splits;
    BinaryFunction enum_functor;
public:
    evaluateFilteredDPSub(
        thrust::device_ptr<RelationID> _pending_keys,
        BinaryFunction _enum_functor,
        int _sq,
        int _qss,
        uint64_t _n_pending_sets,
        int _n_splits
    ) : pending_keys(_pending_keys), 
        enum_functor(_enum_functor), sq(_sq), 
        qss(_qss), n_pending_sets(_n_pending_sets), n_splits(_n_splits)
    {}

    __device__
    thrust::tuple<RelationID, JoinRelation>  operator()(uint64_t tid)
    {
        uint64_t rid = n_pending_sets - 1 - (tid / n_splits);
        uint64_t cid = tid % n_splits;
    
        RelationID relid = pending_keys[rid];

        LOG_DEBUG("[%llu] n_splits=%d, rid=%llu, cid=%llu, relid=%llu\n", 
                tid, n_splits, rid, cid, relid);
        
        JoinRelation jr_out = enum_functor(relid, cid);
        Assert(jr_out.id == BMS64_EMPTY || jr_out.id == relid);
        return thrust::make_tuple<RelationID, JoinRelation>(relid, jr_out);
    }
};

int dpsub_filtered_iteration(int iter, dpsub_iter_param_t &params){   
    int n_iters = 0;
    uint64_t set_offset = 0;
    uint64_t n_pending_sets = 0;
    while (set_offset < params.n_sets){
        uint64_t n_remaining_sets = params.n_sets - set_offset;
        
        while(n_pending_sets < gpuqo_dpsub_n_parallel
                && n_remaining_sets > 0){
            uint64_t n_tab_sets;

            if (n_remaining_sets > PENDING_KEYS_SIZE-n_pending_sets){
                n_tab_sets = PENDING_KEYS_SIZE-n_pending_sets;
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
                uint64_t n_valid_relids = 0;
                for (uint64_t sid=0; sid < n_tab_sets; sid++){
                    RelationID relid = dpsub_unrank_sid(sid, iter, params.info->n_rels, params.binoms.data()) << 1;
                    if (is_connected(relid, params.info)){
                        relids[n_valid_relids++] = relid; 
                    }
                }
                thrust::copy(relids.begin(), relids.begin()+n_valid_relids, params.gpu_pending_keys.begin()+n_pending_sets);

                n_pending_sets += n_valid_relids;
                STOP_TIMING(unrank);
            } else {
                // fill pending keys and filter on GPU 
                START_TIMING(unrank);
                thrust::tabulate(
                    params.gpu_pending_keys.begin()+n_pending_sets,
                    params.gpu_pending_keys.begin()+(n_pending_sets+n_tab_sets),
                    unrankFilteredDPSub(
                        params.info->n_rels,
                        params.gpu_binoms.data(),
                        iter,
                        set_offset
                    ) 
                );
                STOP_TIMING(unrank);

                START_TIMING(filter);
                auto keys_end_iter = thrust::remove_if(
                    params.gpu_pending_keys.begin()+n_pending_sets,
                    params.gpu_pending_keys.begin()+(n_pending_sets+n_tab_sets),
                    filterDisconnectedRelations(params.info)
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

        uint64_t n_joins_per_thread;
        uint64_t n_sets_per_iteration;
        uint64_t threads_per_set;
        uint64_t factor = gpuqo_dpsub_n_parallel / n_pending_sets;

        if (factor < 32 || params.n_joins_per_set <= 32){
            threads_per_set = 32;
        } else{
            threads_per_set = BMS64_HIGHEST(min(factor, params.n_joins_per_set));
        }
        
        n_joins_per_thread = ceil_div(params.n_joins_per_set, threads_per_set);
        n_sets_per_iteration = min(gpuqo_dpsub_n_parallel / threads_per_set, n_pending_sets);

        LOG_PROFILE("n_joins_per_thread=%llu, n_sets_per_iteration=%llu, threads_per_set=%llu, factor=%llu\n",
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
        while (n_pending_sets >= gpuqo_dpsub_n_parallel 
            || (n_pending_sets > 0 && n_remaining_sets == 0)
        ){
            uint64_t n_eval_sets = min(n_sets_per_iteration, n_pending_sets);
            uint64_t n_threads = n_eval_sets * threads_per_set;

            START_TIMING(compute);
            if (use_csg) {
                thrust::tabulate(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        params.gpu_scratchpad_keys.begin(),
                        params.gpu_scratchpad_vals.begin()
                    )),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        params.gpu_scratchpad_keys.begin()+n_threads,
                        params.gpu_scratchpad_vals.begin()+n_threads
                    )),
                    evaluateFilteredDPSub<dpsubEnumerateCsg>(
                        params.gpu_pending_keys.data(),
                        dpsubEnumerateCsg(
                            params.gpu_memo_vals.data(),
                            params.info,
                            threads_per_set
                        ),
                        params.info->n_rels,
                        iter,
                        n_pending_sets,
                        threads_per_set
                    )                
                );
            } else {
                thrust::tabulate(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        params.gpu_scratchpad_keys.begin(),
                        params.gpu_scratchpad_vals.begin()
                    )),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        params.gpu_scratchpad_keys.begin()+n_threads,
                        params.gpu_scratchpad_vals.begin()+n_threads
                    )),
                    evaluateFilteredDPSub<dpsubEnumerateAllSubs>(
                        params.gpu_pending_keys.data(),
                        dpsubEnumerateAllSubs(
                            params.gpu_memo_vals.data(),
                            params.info,
                            threads_per_set
                        ),
                        params.info->n_rels,
                        iter,
                        n_pending_sets,
                        threads_per_set
                    )             
                );
            }            
            STOP_TIMING(compute);

            LOG_DEBUG("After tabulate\n");
            DUMP_VECTOR(params.gpu_scratchpad_keys.begin(), params.gpu_scratchpad_keys.begin()+n_threads);
            DUMP_VECTOR(params.gpu_scratchpad_vals.begin(), params.gpu_scratchpad_vals.begin()+n_threads);

            dpsub_prune_scatter(threads_per_set, n_threads, params);

            n_pending_sets -= n_eval_sets;
        }

        n_iters++;
    }

    return n_iters;
}
