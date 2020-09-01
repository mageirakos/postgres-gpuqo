/*-------------------------------------------------------------------------
 *
 * gpuqo_dpsub.cuh
 *	  declaration of gpuqo DPsub related functions and functors.
 *
 * src/include/optimizer/gpuqo_dpsub.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_CUH
#define GPUQO_DPSUB_CUH

#include <thrust/tabulate.h>

#define PENDING_KEYS_SIZE (gpuqo_dpsub_n_parallel*gpuqo_dpsub_filter_keys_overprovisioning)

__host__ __device__
RelationID dpsub_unrank_sid(uint64_t sid, uint64_t qss, uint64_t sq, uint64_t* binoms);

__device__
void try_join(RelationID relid, JoinRelation &jr_out, 
            RelationID l, RelationID r, JoinRelation* memo_vals,
            BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table);

typedef struct dpsub_iter_param_t{
    BaseRelation *base_rels;
    int n_rels;
    EdgeInfo *edge_table;
    RelationID out_relid;
    thrust::device_vector<BaseRelation> gpu_base_rels;
    thrust::device_vector<EdgeInfo> gpu_edge_table;
    thrust::device_vector<JoinRelation> gpu_memo_vals;
    thrust::host_vector<uint64_t> binoms;
    thrust::device_vector<uint64_t> gpu_binoms;
    uninit_device_vector_relid gpu_pending_keys;
    uninit_device_vector_relid gpu_scratchpad_keys;
    uninit_device_vector_joinrel gpu_scratchpad_vals;
    uninit_device_vector_relid gpu_reduced_keys;
    uninit_device_vector_joinrel gpu_reduced_vals;
    uint64_t n_sets;
    uint64_t n_joins_per_set;
    uint64_t tot;
} dpsub_iter_param_t;

typedef thrust::pair<uninit_device_vector_relid::iterator, uninit_device_vector_joinrel::iterator> scatter_iter_t;

typedef thrust::binary_function<RelationID, uint64_t, JoinRelation> pairs_enum_func_t;

int dpsub_unfiltered_iteration(int iter, dpsub_iter_param_t &params);
int dpsub_filtered_iteration(int iter, dpsub_iter_param_t &params);

void dpsub_prune_scatter(int threads_per_set, int n_threads, dpsub_iter_param_t &params);

struct dpsubEnumerateAllSubs : public pairs_enum_func_t 
{
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    thrust::device_ptr<EdgeInfo> edge_table;
    int sq;
    int n_splits;
public:
    dpsubEnumerateAllSubs(
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        int _sq,
        thrust::device_ptr<EdgeInfo> _edge_table,
        int _n_splits
    ) : memo_vals(_memo_vals), base_rels(_base_rels), sq(_sq), 
        edge_table(_edge_table), n_splits(_n_splits)
    {}

    __device__
    JoinRelation operator()(RelationID relid, uint64_t cid);
};

struct dpsubEnumerateCsg : public pairs_enum_func_t 
{
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    thrust::device_ptr<EdgeInfo> edge_table;
    int sq;
    int n_splits;
public:
    dpsubEnumerateCsg(
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        int _sq,
        thrust::device_ptr<EdgeInfo> _edge_table,
        int _n_splits
    ) : memo_vals(_memo_vals), base_rels(_base_rels), sq(_sq), 
        edge_table(_edge_table), n_splits(_n_splits)
    {}

    __device__
    JoinRelation operator()(RelationID relid, uint64_t cid);
};

EXTERN_PROTOTYPE_TIMING(unrank);
EXTERN_PROTOTYPE_TIMING(filter);
EXTERN_PROTOTYPE_TIMING(compute);
EXTERN_PROTOTYPE_TIMING(prune);
EXTERN_PROTOTYPE_TIMING(scatter);

#endif							/* GPUQO_DPSUB_CUH */
