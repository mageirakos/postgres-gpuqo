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
#include "gpuqo_dpsub_filtered_kernels.cuh"

// user-configured variables (generic)
bool gpuqo_dpsub_filter_enable;
int gpuqo_dpsub_filter_threshold;
int gpuqo_dpsub_filter_cpu_enum_threshold;
int gpuqo_dpsub_filter_keys_overprovisioning;

// user-configured variables (ccc)
bool gpuqo_dpsub_ccc_enable;

// user-configured variables (csg)
bool gpuqo_dpsub_csg_enable;
int gpuqo_dpsub_csg_threshold;

// user-configured variables (tree)
bool gpuqo_dpsub_tree_enable;

// user-configured variables (bicc)
bool gpuqo_dpsub_bicc_enable;


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
                graph_pending = dpsub_csg_evaluation(
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
            n_pending_sets = dpsub_csg_evaluation(
                                        iter, n_remaining_sets, 
                                           0, n_pending_sets, params);
        }
        
        n_iters++;
    }

    return n_iters;
}

template int dpsub_filtered_iteration<Bitmapset32>(int iter, dpsub_iter_param_t<Bitmapset32> &params);
template int dpsub_filtered_iteration<Bitmapset64>(int iter, dpsub_iter_param_t<Bitmapset64> &params);