/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpccp.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpccp.cu
 *
 *-------------------------------------------------------------------------
 */

#include <list>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_sequential.cuh"
#include "gpuqo_cpu_dpe.cuh"

template<typename BitmapsetN, typename memo_t>
class DPccpCPUAlgorithm : public CPUAlgorithm<BitmapsetN, memo_t>{
private:

    void enumerate_csg_rec(BitmapsetN S, BitmapsetN X, BitmapsetN cmp){
        LOG_DEBUG("enumerate_csg_rec(%u, %u, %u)\n", S.toUint(), X.toUint(), cmp.toUint());
        auto info = CPUAlgorithm<BitmapsetN, memo_t>::info;
        BitmapsetN N = get_neighbours(S, info->edge_table) - X;
        if (N.empty())
            return;

        BitmapsetN subset = N.lowest();
        while (!subset.empty()){
            emit(cmp, S | subset);
            subset = nextSubset(subset, N);
        }

        subset = N.lowest();
        while (!subset.empty()){
            enumerate_csg_rec(S|subset, X|N, cmp);
            subset = nextSubset(subset, N);
        }
    }

    void enumerate_csg(){
        auto info = CPUAlgorithm<BitmapsetN, memo_t>::info;
        for (int i=info->n_rels; i>=1; i--){
            BitmapsetN subset = BitmapsetN::nth(i);

            emit(subset, BitmapsetN(0));
            enumerate_csg_rec(subset, subset.allLowerInc(), BitmapsetN(0));
        }
    }

    void enumerate_cmp(BitmapsetN S){
        LOG_DEBUG("enumerate_cmp(%u)\n", S.toUint());
        auto info = CPUAlgorithm<BitmapsetN, memo_t>::info;

        BitmapsetN X = S.allLowerInc();
        BitmapsetN N = get_neighbours(S, info->edge_table) - X;
        BitmapsetN temp = N;

        while (!temp.empty()){
            BitmapsetN v = temp.highest();

            emit(S, v);

            BitmapsetN newX = X | (v.allLowerInc() & N);
            enumerate_csg_rec(v, newX, S);
            
            temp ^= v;
        }
    }

    void emit(BitmapsetN left_id, BitmapsetN right_id){
        LOG_DEBUG("gpuqo_cpu_dpccp_emit(%u, %u)\n", left_id.toUint(), right_id.toUint());
        auto &memo = *CPUAlgorithm<BitmapsetN, memo_t>::memo;

        if (!left_id.empty() && !right_id.empty()){
            auto left = memo.find(left_id);
            auto right = memo.find(right_id);

            Assert(left != memo.end() && right != memo.end());

            JoinRelationCPU<BitmapsetN> *left_rel = left->second;
            JoinRelationCPU<BitmapsetN> *right_rel = right->second;
            BitmapsetN joinset = left_id | right_id;
            int level = joinset.size();

#ifdef GPUQO_PRINT_N_JOINS
            CPUAlgorithm<BitmapsetN, memo_t>::n_checks++;
#endif
            (*CPUAlgorithm<BitmapsetN, memo_t>::join)(level, true, *right_rel, *left_rel);

        } else if (!left_id.empty()) {
            enumerate_cmp(left_id);
        } else{
            enumerate_cmp(right_id);
        }
    }

public:

    virtual void enumerate()
    {
        auto info = CPUAlgorithm<BitmapsetN, memo_t>::info;
        
        if (info->n_iters == info->n_rels)
            enumerate_csg();
        else
            printf("ERROR! IDP does not work with DPccp\n");
    }

    virtual bool check_join(int level, 
        JoinRelationCPU<BitmapsetN> &left_rel, 
        JoinRelationCPU<BitmapsetN>&right_rel)
    {      
        // No check is necessary since dpccp guarantees all joinpairs are valid
        Assert(is_disjoint_rel(left_rel, right_rel) 
            && are_connected_rel(left_rel, right_rel, 
                CPUAlgorithm<BitmapsetN, memo_t>::info));
        return true;
    }
};

/* gpuqo_cpu_dpccp
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_dpccp(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPccpCPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN> > alg;
    return gpuqo_cpu_sequential(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_cpu_dpccp<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_dpccp<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);

/* gpuqo_dpe_dpccp
 *
 *	 Parallel CPU baseline for GPU query optimization using the DP size
 *   algorithm.
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_dpe_dpccp(GpuqoPlannerInfo<BitmapsetN>* info)
{
    DPccpCPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN> > alg;
    return gpuqo_cpu_dpe(info, &alg);
}

template QueryTree<Bitmapset32>* gpuqo_dpe_dpccp<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_dpe_dpccp<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
