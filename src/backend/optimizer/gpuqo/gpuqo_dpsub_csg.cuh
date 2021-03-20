/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_csg.cuh
 *      "CSG" enumeration functor for DPsub
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_csg.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_ENUM_CSG_CUH
#define GPUQO_DPSUB_ENUM_CSG_CUH

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_dpsub.cuh"

struct dpsubEnumerateCsg : public pairs_enum_func_t 
{
    HashTable32bit memo;
    GpuqoPlannerInfo* info;
    int n_splits;
public:
    dpsubEnumerateCsg(
        HashTable32bit _memo,
        GpuqoPlannerInfo* _info,
        int _n_splits
    ) : memo(_memo), info(_info), n_splits(_n_splits)
    {}

    __device__
    JoinRelation operator()(RelationID relid, uint32_t cid);
};

#endif              // GPUQO_DPSUB_ENUM_ALL_SUBS_CUH