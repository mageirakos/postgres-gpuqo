/*------------------------------------------------------------------------
 *
 * gpuqo_dpsize.c
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsize.c
 *
 *-------------------------------------------------------------------------
 */

#include <iostream>
#include <cmath>

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

#define printVectorOffset(from, offset, to) { \
    auto mIter = (from); \
    mIter += (offset); \
    for(int mCount=offset; mIter != (to); ++mIter, ++mCount) \
        std::cout << mCount << " : " << *mIter << std::endl; \
}

#define printVector(from, to) printVectorOffset((from), 0, (to))

struct enumerate : public thrust::unary_function< int,thrust::tuple<RelationID, JoinRelation> >
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<unsigned int> partition_offsets;
    thrust::device_ptr<unsigned int> partition_sizes;
    int iid;
public:
    enumerate(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<unsigned int> _partition_offsets,
        thrust::device_ptr<unsigned int> _partition_sizes,
        int _iid
    ) : memo_keys(_memo_keys), partition_offsets(_partition_offsets), 
    partition_sizes(_partition_sizes), iid(_iid)
    {}

    __device__
    thrust::tuple<RelationID, JoinRelation> operator()(unsigned int cid) 
    {
        int lp = 0;
        int rp = iid - 2;
        int o = partition_sizes[lp] * partition_sizes[rp];

        while (cid >= o){
            cid -= o;
            lp++;
            rp--;
            o = partition_sizes[lp] * partition_sizes[rp];
        }

        int l = cid / partition_sizes[rp];
        int r = cid % partition_sizes[rp];

        RelationID relid;
        JoinRelation jr;

        jr.left_relation_idx = partition_offsets[lp] + l;
        jr.right_relation_idx = partition_offsets[rp] + r;

        relid = memo_keys[jr.left_relation_idx] | memo_keys[jr.right_relation_idx];

        return thrust::tuple<RelationID, JoinRelation>(relid, jr);
    }
};


struct filter : public thrust::unary_function<thrust::tuple<RelationID, JoinRelation>, bool>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    int n_rels;
public:
    filter(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        int _n_rels
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), base_rels(_base_rels),
        n_rels(_n_rels)
    {}

    __device__
    bool operator()(thrust::tuple<RelationID, JoinRelation> t) 
    {
        RelationID relid = t.get<0>();
        JoinRelation jr = t.get<1>();

        RelationID left_id = memo_keys[jr.left_relation_idx];
        RelationID right_id = memo_keys[jr.right_relation_idx];
        JoinRelation left_rel = memo_vals[jr.left_relation_idx];
        JoinRelation right_rel = memo_vals[jr.right_relation_idx];

        if (left_id & right_id) // not disjoint
            return true;

        RelationID left_edges = 0;
        for (int i = 0; i < n_rels; i++){
            int base_relid = 1<<i;
            BaseRelation base_rel = base_rels[i];
            if (left_id & base_relid){
                left_edges |= base_rel.edges;
            }
        }

        if (left_edges & right_id) // connected
            return false;
        else // not connected
            return true;
    }
};

struct cost : public thrust::unary_function<JoinRelation,JoinRelation>
{
    thrust::device_ptr<RelationID> memo_keys;
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    int n_rels;
public:
    cost(
        thrust::device_ptr<RelationID> _memo_keys,
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        int _n_rels
    ) : memo_keys(_memo_keys), memo_vals(_memo_vals), base_rels(_base_rels),
        n_rels(_n_rels)
    {}

    __device__
    JoinRelation operator()(JoinRelation jr) 
    {
        RelationID left_id = memo_keys[jr.left_relation_idx];
        RelationID right_id = memo_keys[jr.right_relation_idx];
        JoinRelation left_rel = memo_vals[jr.left_relation_idx];
        JoinRelation right_rel = memo_vals[jr.right_relation_idx];

        double sel = 1.0;
        
        // maybe I can first accumulate all edges as in filter?
        // maybe I can iterate more efficiently in theese bitsets
        for (int i = 0; i < n_rels; i++){
            int base_relid = 1<<i;
            if (left_id & base_relid){
                for (int j = 0; j < n_rels; j++){
                    int peer_relid = 1<<j;
                    BaseRelation base_rel_left = base_rels[i];
                    if (base_rel_left.edges & peer_relid & right_id){
                        BaseRelation base_rel_right = base_rels[j];
                        sel *= 1.0 / base_rel_right.tuples;
                    }
                }
            }
        }
        
        double rows = sel * (double) left_rel.rows * (double) right_rel.rows;
        jr.rows = rows > 1 ? round(rows) : 1;
        jr.cost = jr.rows + left_rel.cost + right_rel.cost;

        return jr;
    }
};

void buildQueryTree(int idx, 
                            thrust::device_vector<RelationID> &gpu_memo_keys,
                            thrust::device_vector<JoinRelation> &gpu_memo_vals,
                            QueryTree **qt)
{
    JoinRelation jr = gpu_memo_vals[idx];
    RelationID relid = gpu_memo_keys[idx];

    (*qt) = (QueryTree*) malloc(sizeof(QueryTree));
    (*qt)->id = relid;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr.rows;
    (*qt)->cost = jr.cost;

    if (jr.left_relation_idx == 0 && jr.right_relation_idx == 0)
        return;

    buildQueryTree(jr.left_relation_idx, gpu_memo_keys, gpu_memo_vals, &((*qt)->left));
    buildQueryTree(jr.right_relation_idx, gpu_memo_keys, gpu_memo_vals, &((*qt)->right));
}

/* gpuqo_dpsize
 *
 *	 GPU query optimization using the DP size variant.
 */
extern "C"
QueryTree*
gpuqo_dpsize(BaseRelation baserels[], int N)
{
    DECLARE_TIMING(gpuqo_dpsize);
    DECLARE_TIMING(init);
    DECLARE_TIMING(execute);
    
    START_TIMING(gpuqo_dpsize);
    START_TIMING(init);
    
    thrust::device_vector<BaseRelation> gpu_baserels(baserels, baserels + N);
    thrust::device_vector<RelationID> gpu_memo_keys(std::pow(2,N));
    thrust::device_vector<JoinRelation> gpu_memo_vals(std::pow(2,N));
    thrust::host_vector<unsigned int> partition_offsets(N);
    thrust::host_vector<unsigned int> partition_sizes(N);
    thrust::device_vector<unsigned int> gpu_partition_offsets(N);
    thrust::device_vector<unsigned int> gpu_partition_sizes(N);
    QueryTree* out = NULL;

    for(int i=0; i<N; i++){
        gpu_memo_keys[i] = baserels[i].id;
        JoinRelation t;
        t.left_relation_idx = 0; 
        t.right_relation_idx = 0; 
        t.cost = 0.2*baserels[i].rows; 
        t.rows = baserels[i].rows; 
        gpu_memo_vals[i] = t;
        partition_sizes[i] = i == 0 ? N : 0;
        partition_offsets[i] = i == 1 ? N : 0;
    }
    gpu_partition_offsets = partition_offsets;
    gpu_partition_sizes = partition_sizes;

    STOP_TIMING(init);

#ifdef GPUQO_DEBUG
    printVector(gpu_memo_keys.begin(), gpu_memo_keys.begin() + N);
    printVector(gpu_memo_vals.begin(), gpu_memo_vals.begin() + N);    
#endif

    START_TIMING(execute);
    try{
        DECLARE_TIMING(iter_init);
        DECLARE_TIMING(enumerate);
        DECLARE_TIMING(filter);
        DECLARE_TIMING(sort);
        DECLARE_TIMING(compute_prune);
        DECLARE_TIMING(update_offsets);
        DECLARE_TIMING(build_qt);

        for(int i=2; i<=N; i++){
            START_TIMING(iter_init);
            // calculate size of required temp space
            int n_combinations = 0;
            for (int j=1; j<i; j++){
                n_combinations += partition_sizes[j-1] * partition_sizes[i-j-1];
            }

#ifdef GPUQO_DEBUG
            printf("Starting iteration %d: %d combinations\n", i, n_combinations);
#endif
            // allocate temp scratchpad
            thrust::device_vector<RelationID> gpu_scratchpad_keys(n_combinations);
            thrust::device_vector<JoinRelation> gpu_scratchpad_vals(n_combinations);

            STOP_TIMING(iter_init);
            START_TIMING(enumerate);
            
            // fill scratchpad
            thrust::tabulate(
                thrust::make_zip_iterator(thrust::make_tuple(
                    gpu_scratchpad_keys.begin(),
                    gpu_scratchpad_vals.begin()
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    gpu_scratchpad_keys.end(),
                    gpu_scratchpad_vals.end()
                )),
                enumerate(
                    gpu_memo_keys.data(), 
                    gpu_partition_offsets.data(), 
                    gpu_partition_sizes.data(), 
                    i
                )
            );

            STOP_TIMING(enumerate);

#ifdef GPUQO_DEBUG
            printf("After tabulate\n");
            printVector(gpu_scratchpad_keys.begin(), gpu_scratchpad_keys.end());
            printVector(gpu_scratchpad_vals.begin(), gpu_scratchpad_vals.end());
#endif

            START_TIMING(filter);
            // filter out invalid pairs
            auto newEnd = thrust::remove_if(
                thrust::make_zip_iterator(thrust::make_tuple(
                    gpu_scratchpad_keys.begin(),
                    gpu_scratchpad_vals.begin()
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    gpu_scratchpad_keys.end(),
                    gpu_scratchpad_vals.end()
                )),
                filter(
                    gpu_memo_keys.data(), 
                    gpu_memo_vals.data(),
                    gpu_baserels.data(), 
                    N
                )
            );

            STOP_TIMING(filter);

#ifdef GPUQO_DEBUG
            printf("After remove_if\n");
            printVector(gpu_scratchpad_keys.begin(), newEnd.get_iterator_tuple().get<0>());
            printVector(gpu_scratchpad_vals.begin(), newEnd.get_iterator_tuple().get<1>());
#endif

            START_TIMING(sort);

            // sort by key (prepare for pruning)
            thrust::sort_by_key(
                gpu_scratchpad_keys.begin(),
                newEnd.get_iterator_tuple().get<0>(),
                gpu_scratchpad_vals.begin()
            );

            STOP_TIMING(sort);

#ifdef GPUQO_DEBUG
            printf("After sort_by_key\n");
            printVector(gpu_scratchpad_keys.begin(), newEnd.get_iterator_tuple().get<0>());
            printVector(gpu_scratchpad_vals.begin(), newEnd.get_iterator_tuple().get<1>());
#endif

            START_TIMING(compute_prune);

            // calculate cost, prune and copy to table
            auto out_iters = thrust::reduce_by_key(
                gpu_scratchpad_keys.begin(),
                newEnd.get_iterator_tuple().get<0>(),
                thrust::make_transform_iterator(
                    gpu_scratchpad_vals.begin(),
                    cost(
                        gpu_memo_keys.data(), 
                        gpu_memo_vals.data(),
                        gpu_baserels.data(),
                        N
                    )
                ),
                gpu_memo_keys.begin()+partition_offsets[i-1],
                gpu_memo_vals.begin()+partition_offsets[i-1],
                thrust::equal_to<unsigned int>(),
                thrust::minimum<JoinRelation>()
            );

            STOP_TIMING(compute_prune);

#ifdef GPUQO_DEBUG
            printf("After reduce_by_key\n");
            printVector(gpu_memo_keys.begin(), out_iters.first);
            printVector(gpu_memo_vals.begin(), out_iters.second);
#endif

            START_TIMING(update_offsets);

            // update ps and po
            partition_sizes[i-1] = thrust::distance(
                gpu_memo_keys.begin()+partition_offsets[i-1],
                out_iters.first
            ); // TODO check inclusive/exclusive
            gpu_partition_sizes[i-1] = partition_sizes[i-1];
            
            if (i < N){
                partition_offsets[i] = partition_sizes[i-1] + partition_offsets[i-1];
                gpu_partition_offsets[i] = partition_offsets[i];
            }

            STOP_TIMING(update_offsets);

#ifdef GPUQO_DEBUG
            printf("After partition_*\n");
            printVector(partition_sizes.begin(), partition_sizes.end());
            printVector(partition_offsets.begin(), partition_offsets.end());
#endif

            PRINT_TIMING(iter_init);
            PRINT_TIMING(enumerate);
            PRINT_TIMING(filter);
            PRINT_TIMING(sort);
            PRINT_TIMING(compute_prune);
            PRINT_TIMING(update_offsets);
        }

        START_TIMING(build_qt);
            
        buildQueryTree(partition_offsets[N-1], gpu_memo_keys, gpu_memo_vals, &out);
    
        STOP_TIMING(build_qt);
    
        PRINT_TOTAL_TIMING(iter_init);
        PRINT_TOTAL_TIMING(enumerate);
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
