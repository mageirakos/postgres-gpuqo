/*------------------------------------------------------------------------
 *
 * gpuqo_dpsize.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsize.cu
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
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/tuple.h>
#include <thrust/system/system_error.h>
#include <thrust/distance.h>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_query_tree.cuh"

#define KB 1024ULL
#define MB (KB*1024)
#define GB (MB*1024)
#define RELSIZE (sizeof(JoinRelation)+sizeof(RelationID))
#define SCR_ENTRY_SIZE (sizeof(uint2)+sizeof(RelationID))

int gpuqo_scratchpad_size_mb;
int gpuqo_max_memo_size_mb;

/* unrankDPSize
 *
 *	 unrank algorithm for DPsize GPU variant
 */
struct unrankDPSize : public thrust::unary_function< uint32_t,thrust::tuple<RelationID, uint2> >
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<uint32_t> partition_offsets;
    thrust::device_ptr<uint32_t> partition_sizes;
    int iid;
    uint64_t offset;
public:
    unrankDPSize(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<uint32_t> _partition_offsets,
        thrust::device_ptr<uint32_t> _partition_sizes,
        int _iid,
        uint64_t _offset
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), 
        partition_offsets(_partition_offsets), 
        partition_sizes(_partition_sizes), iid(_iid), offset(_offset)
    {}

        __device__
        thrust::tuple<RelationID, uint2> operator()(uint64_t cid) 
        {
            __shared__ uint64_t offsets[32];
            int n_active = __popc(__activemask());

            for (int i = threadIdx.x; i <= iid-2; i += n_active){
                offsets[i] = ((uint64_t)partition_sizes[i]) * partition_sizes[iid-2-i];
            }
            __syncthreads();
    
            uint32_t lp = 0;
            uint32_t rp = iid - 2;
            uint64_t o = offsets[lp];
            cid += offset;

            while (cid >= o){
                cid -= o;
                lp++;
                rp--;

                Assert(lp <= iid-2 && lp < 32);
                Assert(rp <= iid-2 && rp < 32);

                o = offsets[lp];
            }

            uint32_t l = cid / partition_sizes[rp];
            uint32_t r = cid % partition_sizes[rp];
    
            RelationID relid;
            uint2 out = make_uint2(
                partition_offsets[lp] + l, 
                partition_offsets[rp] + r
            );
    
            LOG_DEBUG("%llu: %u %u\n", 
                cid, 
                out.x,
                out.y
            );
    
            relid = BMS32_UNION(memo_keys[out.x], memo_keys[out.y]);
    
            return thrust::tuple<RelationID, uint2>(relid, out);
        }
};


struct filterJoinedDisconnected : public thrust::unary_function<thrust::tuple<RelationID, uint2>, bool>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    GpuqoPlannerInfo* info;
public:
    filterJoinedDisconnected(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        GpuqoPlannerInfo* _info
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), info(_info)
    {}

    __device__
    bool operator()(thrust::tuple<RelationID, uint2> t) 
    {
        RelationID relid = t.get<0>();
        uint2 idxs = t.get<1>();

        LOG_DEBUG("%d %d: %u %u\n", 
            blockIdx.x,
            threadIdx.x,
            idxs.x,
            idxs.y
        );
        
        JoinRelation& left_rel = memo_vals.get()[idxs.x];
        JoinRelation& right_rel = memo_vals.get()[idxs.y];

        if (!is_disjoint(left_rel, right_rel)) // not disjoint
            return true;
        else{
            return !are_connected(left_rel, right_rel, info);
        }
    }
};


struct joinCost : public thrust::unary_function<uint2,JoinRelation>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    GpuqoPlannerInfo* info;
public:
    joinCost(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        GpuqoPlannerInfo* _info
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), info(_info)
    {}

    __device__
    JoinRelation operator()(uint2 idxs){
        JoinRelation jr;

        make_join_rel(jr, 
            idxs.x,
            memo_vals.get()[idxs.x], 
            idxs.y,
            memo_vals.get()[idxs.y], 
            info
        );

        return jr;
    }
};

struct joinRelToUint2 : public thrust::unary_function<thrust::tuple<RelationID,JoinRelation>,thrust::tuple<RelationID,uint2> >
{
public:
    joinRelToUint2() {}

    __device__
    thrust::tuple<RelationID,uint2> operator()(thrust::tuple<RelationID,JoinRelation> t){
        JoinRelation& jr = t.get<1>();
        return thrust::make_tuple(
            t.get<0>(),
            make_uint2(jr.left_relation_idx, jr.right_relation_idx)
        );
    }
};

/* gpuqo_dpsize
 *
 *	 GPU query optimization using the DP size variant.
 */
extern "C"
QueryTree*
gpuqo_dpsize(GpuqoPlannerInfo* info)
{
    DECLARE_TIMING(gpuqo_dpsize);
    DECLARE_NV_TIMING(init);
    DECLARE_NV_TIMING(execute);
    
    START_TIMING(gpuqo_dpsize);
    START_TIMING(init);

    uint32_t scratchpad_size = gpuqo_scratchpad_size_mb * MB / SCR_ENTRY_SIZE;
    // at least 2*gpuqo_n_parallel otherwise it would be very inefficient
    if (scratchpad_size < 2*gpuqo_n_parallel)
        scratchpad_size = 2*gpuqo_n_parallel;

    uint32_t prune_threshold = scratchpad_size - gpuqo_n_parallel;
    uint32_t max_memo_size = gpuqo_max_memo_size_mb * MB / RELSIZE;
    uint32_t memo_size = std::min(1U<<info->n_rels, max_memo_size);

    LOG_PROFILE("Using a scratchpad of size %u (prune threshold: %u)\n", 
        scratchpad_size, prune_threshold);
    
    uninit_device_vector_relid gpu_memo_keys(memo_size);
    uninit_device_vector_joinrel gpu_memo_vals(memo_size);
    thrust::host_vector<uint32_t> partition_offsets(info->n_rels);
    thrust::host_vector<uint32_t> partition_sizes(info->n_rels);
    thrust::device_vector<uint32_t> gpu_partition_offsets(info->n_rels);
    thrust::device_vector<uint32_t> gpu_partition_sizes(info->n_rels);
    QueryTree* out = NULL;

    for(int i=0; i<info->n_rels; i++){
        gpu_memo_keys[i] = info->base_rels[i].id;

        JoinRelation t;
        t.id = info->base_rels[i].id;
        t.left_relation_idx = 0; 
        t.left_relation_id = 0; 
        t.right_relation_idx = 0; 
        t.right_relation_id = 0; 
        t.cost = baserel_cost(info->base_rels[i]); 
        t.rows = info->base_rels[i].rows; 
        t.edges = info->edge_table[i];
        gpu_memo_vals[i] = t;

        partition_sizes[i] = i == 0 ? info->n_rels : 0;
        partition_offsets[i] = i == 1 ? info->n_rels : 0;
    }
    gpu_partition_offsets = partition_offsets;
    gpu_partition_sizes = partition_sizes;

    // scratchpad size is increased on demand, starting from a minimum capacity
    uninit_device_vector_relid gpu_scratchpad_keys(scratchpad_size);
    uninit_device_vector_uint2 gpu_scratchpad_vals(scratchpad_size);

    STOP_TIMING(init);

    DUMP_VECTOR(gpu_memo_keys.begin(), gpu_memo_keys.begin() + info->n_rels);
    DUMP_VECTOR(gpu_memo_vals.begin(), gpu_memo_vals.begin() + info->n_rels);    

    START_TIMING(execute);
    try{ // catch any exception in thrust
        DECLARE_NV_TIMING(iter_init);
        DECLARE_NV_TIMING(copy_pruned);
        DECLARE_NV_TIMING(unrank);
        DECLARE_NV_TIMING(filter);
        DECLARE_NV_TIMING(sort);
        DECLARE_NV_TIMING(compute_prune);
        DECLARE_NV_TIMING(update_offsets);
        DECLARE_NV_TIMING(iteration);
        DECLARE_NV_TIMING(build_qt);

        // iterate over the size of the resulting joinrel
        for(int i=2; i<=info->n_rels; i++){
            START_TIMING(iteration);
            START_TIMING(iter_init);
            
            // calculate number of combinations of relations that make up 
            // a joinrel of size i
            uint64_t n_combinations = 0;
            for (int j=1; j<i; j++){
                n_combinations += ((uint64_t)partition_sizes[j-1]) * partition_sizes[i-j-1];
            }

            LOG_PROFILE("\nStarting iteration %d: %llu combinations (scratchpad: %llu)\n", i, n_combinations, scratchpad_size);

            STOP_TIMING(iter_init);

            // offset of cid for the enumeration
            uint64_t offset = 0;

            // count how many times I had to sort+prune
            // NB: either a really high number of tables or a small scratchpad
            //     is required to make multiple sort+prune steps
            int pruning_iter = 0;

            // size of the already filtered joinrels at the beginning of the 
            // scratchpad
            uint32_t temp_size;

            // until I go through every possible iteration
            while (offset < n_combinations){
                if (pruning_iter != 0){
                    // if I already pruned at least once, I need to fetch those 
                    // partially pruned samples from the memo to the scratchpad
                    START_TIMING(copy_pruned);
                    thrust::transform(
                        thrust::make_zip_iterator(thrust::make_tuple(
                            gpu_memo_keys.begin()+partition_offsets[i-1],
                            gpu_memo_vals.begin()+partition_offsets[i-1]
                        )),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            gpu_memo_keys.begin()+partition_offsets[i-1]+partition_sizes[i-1],
                            gpu_memo_vals.begin()+partition_offsets[i-1]+partition_sizes[i-1]
                        )),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            gpu_scratchpad_keys.begin(),
                            gpu_scratchpad_vals.begin()
                        )),
                        joinRelToUint2()
                    );

                    // I now have already ps[i-1] joinrels in the sratchpad
                    temp_size = partition_sizes[i-1];
                    STOP_TIMING(copy_pruned);
                } else {
                    // scratchpad is initially empty
                    temp_size = 0;
                }

                // until there are no more combinations or the scratchpad is
                // too full (I would need to use fewer than `n_parallel` 
                // threads in order to continue)
                while (offset < n_combinations 
                            && temp_size < prune_threshold)
                {
                    // how many combinations I will try at this iteration
                    uint32_t chunk_size;

                    if (n_combinations - offset < scratchpad_size - temp_size){
                        // all remaining
                        chunk_size = n_combinations - offset;
                    } else{
                        // up to scratchpad capacity
                        chunk_size = scratchpad_size - temp_size;
                    }

                    // give possibility to user to interrupt
                    CHECK_FOR_INTERRUPTS();

                    START_TIMING(unrank);
                    // fill scratchpad
                    thrust::tabulate(
                        thrust::make_zip_iterator(thrust::make_tuple(
                            gpu_scratchpad_keys.begin()+temp_size,
                            gpu_scratchpad_vals.begin()+temp_size
                        )),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            gpu_scratchpad_keys.begin()+(chunk_size+temp_size),
                            gpu_scratchpad_vals.begin()+(chunk_size+temp_size)
                        )),
                        unrankDPSize(
                            gpu_memo_keys.data(), 
                            gpu_memo_vals.data(), 
                            gpu_partition_offsets.data(), 
                            gpu_partition_sizes.data(), 
                            i,
                            offset
                        )
                    );
                    STOP_TIMING(unrank);

                    LOG_DEBUG("After tabulate\n");
                    DUMP_VECTOR(
                        gpu_scratchpad_keys.begin()+temp_size, 
                        gpu_scratchpad_keys.begin()+(temp_size+chunk_size)
                    );
                    DUMP_VECTOR(
                        gpu_scratchpad_vals.begin()+temp_size, 
                        gpu_scratchpad_vals.begin()+(temp_size+chunk_size)
                    );

                    // give possibility to user to interrupt
                    CHECK_FOR_INTERRUPTS();

                    START_TIMING(filter);
                    // filter out invalid pairs
                    auto newEnd = thrust::remove_if(
                        thrust::make_zip_iterator(thrust::make_tuple(
                            gpu_scratchpad_keys.begin()+temp_size,
                            gpu_scratchpad_vals.begin()+temp_size
                        )),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            gpu_scratchpad_keys.begin()+(temp_size+chunk_size),
                            gpu_scratchpad_vals.begin()+(temp_size+chunk_size)
                        )),
                        filterJoinedDisconnected(
                            gpu_memo_keys.data(), 
                            gpu_memo_vals.data(),
                            info
                        )
                    );

                    STOP_TIMING(filter);

                    LOG_DEBUG("After remove_if\n");
                    DUMP_VECTOR(
                        gpu_scratchpad_keys.begin()+temp_size, 
                        newEnd.get_iterator_tuple().get<0>()
                    );
                    DUMP_VECTOR(
                        gpu_scratchpad_vals.begin()+temp_size, 
                        newEnd.get_iterator_tuple().get<1>()
                    );

                    // get how many rels remain after filter and add it to 
                    // temp_size
                    temp_size += thrust::distance(
                        gpu_scratchpad_keys.begin()+temp_size,
                        newEnd.get_iterator_tuple().get<0>()
                    );

                    // increase offset for next iteration and/or for exit check 
                    offset += chunk_size;
                } // filtering loop: while(off<n_comb && temp_s < cap)

                // I now have either all valid combinations or `prune_thresh` 
                // combinations. In the first case, I will prune once
                // and exit. In the second case, there are still some 
                // combinations to try so I will prune this partial result and
                // then reload it back in the scratchpad to finish execution

                // give possibility to user to interrupt
                CHECK_FOR_INTERRUPTS();

                START_TIMING(sort);

                // sort by key (prepare for pruning)
                thrust::sort_by_key(
                    gpu_scratchpad_keys.begin(),
                    gpu_scratchpad_keys.begin() + temp_size,
                    gpu_scratchpad_vals.begin()
                );
    
                STOP_TIMING(sort);
    
                LOG_DEBUG("After sort_by_key\n");
                DUMP_VECTOR(gpu_scratchpad_keys.begin(), gpu_scratchpad_keys.begin() + temp_size);
                DUMP_VECTOR(gpu_scratchpad_vals.begin(), gpu_scratchpad_vals.begin() + temp_size);

                // give possibility to user to interrupt
                CHECK_FOR_INTERRUPTS();
    
                START_TIMING(compute_prune);
    
                // calculate cost, prune and copy to table
                auto out_iters = thrust::reduce_by_key(
                    gpu_scratchpad_keys.begin(),
                    gpu_scratchpad_keys.begin() + temp_size,
                    thrust::make_transform_iterator(
                        gpu_scratchpad_vals.begin(),
                        joinCost(
                            gpu_memo_keys.data(), 
                            gpu_memo_vals.data(),
                            info
                        )
                    ),
                    gpu_memo_keys.begin()+partition_offsets[i-1],
                    gpu_memo_vals.begin()+partition_offsets[i-1],
                    thrust::equal_to<RelationID>(),
                    thrust::minimum<JoinRelation>()
                );
    
                STOP_TIMING(compute_prune);
    
                LOG_DEBUG("After reduce_by_key\n");
                DUMP_VECTOR(gpu_memo_keys.begin(), out_iters.first);
                DUMP_VECTOR(gpu_memo_vals.begin(), out_iters.second);
    
                START_TIMING(update_offsets);
    
                // update ps and po
                partition_sizes[i-1] = thrust::distance(
                    gpu_memo_keys.begin()+partition_offsets[i-1],
                    out_iters.first
                );
                gpu_partition_sizes[i-1] = partition_sizes[i-1];
                
                if (i < info->n_rels){
                    partition_offsets[i] = partition_sizes[i-1] + partition_offsets[i-1];
                    gpu_partition_offsets[i] = partition_offsets[i];
                }
    
                STOP_TIMING(update_offsets);
    
                LOG_DEBUG("After partition_*\n");
                DUMP_VECTOR(partition_sizes.begin(), partition_sizes.end());
                DUMP_VECTOR(partition_offsets.begin(), partition_offsets.end());
                
                pruning_iter++;
            } // pruning loop: while(offset<n_combinations)

            STOP_TIMING(iteration);
            LOG_DEBUG("It took %d pruning iterations", pruning_iter);
            
            PRINT_CHECKPOINT_TIMING(iter_init);
            PRINT_CHECKPOINT_TIMING(copy_pruned);
            PRINT_CHECKPOINT_TIMING(unrank);
            PRINT_CHECKPOINT_TIMING(filter);
            PRINT_CHECKPOINT_TIMING(sort);
            PRINT_CHECKPOINT_TIMING(compute_prune);
            PRINT_CHECKPOINT_TIMING(update_offsets);
            PRINT_TIMING(iteration);
        } // dpsize loop: for i = 2..n_rels

        START_TIMING(build_qt);
            
        buildQueryTree(partition_offsets[info->n_rels-1], gpu_memo_vals, &out);
    
        STOP_TIMING(build_qt);
    
        PRINT_TOTAL_TIMING(iter_init);
        PRINT_TOTAL_TIMING(copy_pruned);
        PRINT_TOTAL_TIMING(unrank);
        PRINT_TOTAL_TIMING(filter);
        PRINT_TOTAL_TIMING(sort);
        PRINT_TOTAL_TIMING(compute_prune);
        PRINT_TOTAL_TIMING(update_offsets);
        PRINT_TOTAL_TIMING(build_qt);
    } catch(thrust::system_error err){
        printf("Thrust %d: %s", err.code().value(), err.what());
    }

    STOP_TIMING(execute);
    STOP_TIMING(gpuqo_dpsize);

    PRINT_TIMING(gpuqo_dpsize);
    PRINT_TIMING(init);
    PRINT_TIMING(execute);

    return out;
}
