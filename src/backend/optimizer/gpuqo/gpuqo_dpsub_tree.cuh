/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_tree.cuh
 *      enumeration functors for DPsub when subgraph is a tree
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_tree.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_ENUM_TREE_CUH
#define GPUQO_DPSUB_ENUM_TREE_CUH

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_dpsub.cuh"

struct dpsubEnumerateTreeSimple : public pairs_enum_func_t 
{
    HashTable32bit memo;
    GpuqoPlannerInfo* info;
    int n_splits;
public:
    dpsubEnumerateTreeSimple(
        HashTable32bit _memo,
        GpuqoPlannerInfo* _info,
        int _n_splits
    ) : memo(_memo), info(_info), n_splits(_n_splits)
    {}

    __device__
    JoinRelation operator()(RelationID relid, uint32_t cid);
};

struct dpsubEnumerateTreeWithSubtrees : public pairs_enum_func_t 
{
    HashTable32bit memo;
    GpuqoPlannerInfo* info;
    int n_splits;
public:
    dpsubEnumerateTreeWithSubtrees(
        HashTable32bit _memo,
        GpuqoPlannerInfo* _info,
        int _n_splits
    ) : memo(_memo), info(_info), n_splits(_n_splits)
    {}

    __device__
    JoinRelation operator()(RelationID relid, uint32_t cid);
};

#endif              // GPUQO_DPSUB_ENUM_TREE_CUH