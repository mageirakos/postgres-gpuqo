/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpsub_bicc.cuh
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_bicc.cuh
 *
 *-------------------------------------------------------------------------
 */

 #include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cpu_sequential.cuh"
#include "gpuqo_cpu_level_hashtable.cuh"

template<typename BitmapsetN, typename memo_t, bool manage_best>
class DPsubGenericCPUAlgorithm : public CPUAlgorithm<BitmapsetN, memo_t> {
public:
    virtual JoinRelationCPU<BitmapsetN> *enumerate_subsets(BitmapsetN set)
    {
        return NULL;
    }

    virtual void enumerate()
    {
        auto info = CPUAlgorithm<BitmapsetN, memo_t>::info;
        // first bit is zero
        for (BitmapsetN i=1; i < BitmapsetN::nth(info->n_rels); i++){
            BitmapsetN join_id = i << 1; // first bit is 0 in Postgres

            if (!is_connected(join_id, info->edge_table))
                continue;

            enumerate_subsets(join_id);
        }
    }
};

template<typename BitmapsetN>
using alg_t = DPsubGenericCPUAlgorithm<BitmapsetN, level_hashtable<BitmapsetN>,true>;

template<typename BitmapsetN>
QueryTree<BitmapsetN>* gpuqo_cpu_dpsub_generic_parallel(
                        GpuqoPlannerInfo<BitmapsetN>* info, 
                        alg_t<BitmapsetN> *algorithm);

