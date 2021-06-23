/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dplin.cu
 *      DPlin using IKKBZ
 *
 * src/backend/optimizer/gpuqo/gpuqo_cpu_dplin.cu
 *
 *-------------------------------------------------------------------------
 */

#include <list>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_common.cuh"
#include "gpuqo_query_tree.cuh"

using namespace std;

int gpuqo_dplin_csg_threshold_lindp;
int gpuqo_dplin_threshold_lindp;
int gpuqo_dplin_threshold_idp;

template<typename BitmapsetN>
static int
countCCRec(BitmapsetN S, BitmapsetN X, int c, int budget, 
            GpuqoPlannerInfo<BitmapsetN> *info)
{
    BitmapsetN N = get_neighbours(S, info->edge_table) - X;

    if (N.empty())
        return c;

    BitmapsetN subset = N.lowest();
    while (!subset.empty()){
        if (++c > budget)
            return c;
        c = countCCRec(S|subset, X|N, c, budget, info);
        subset = nextSubset(subset, N);
    }

    return c;
}

template<typename BitmapsetN>
static int
countCC(GpuqoPlannerInfo<BitmapsetN> *info, int budget)
{
    int c = 0;
    for (int i = 0; i < info->n_rels; i++){
        if (++c > budget)
            return c;
        
        BitmapsetN v = BitmapsetN::nth(i+1);
        BitmapsetN B = v.allLower();

        c = countCCRec(v, B, c, budget, info);
    }  

    return c;
}

/* gpuqo_cpu_dplin
 *
 *	 DPlin implementation:
 *   choose between DPccp, linearized DP or IDP2/GOO with linearized DP
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_dplin(GpuqoPlannerInfo<BitmapsetN> *info)
{
    int budget = gpuqo_dplin_csg_threshold_lindp;
    if (info->n_rels < gpuqo_dplin_threshold_lindp 
            || countCC(info, budget) <= budget){
        LOG_PROFILE("dplin: using dpccp\n");
        return gpuqo_cpu_dpccp(info);
    } else if (info->n_rels < gpuqo_dplin_threshold_idp) {
        LOG_PROFILE("dplin: using linearized dp\n");
        return gpuqo_cpu_linearized_dp(info);
    } else {
        LOG_PROFILE("dplin: using goo-dp\n");
        return gpuqo_run_idp2(GPUQO_CPU_LINEARIZED_DP, info, gpuqo_dplin_threshold_idp);
    }
}

template QueryTree<Bitmapset32>* gpuqo_cpu_dplin<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_dplin<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
template QueryTree<BitmapsetDynamic>* gpuqo_cpu_dplin<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic>*);
