/*------------------------------------------------------------------------
 *
 * gpuqo_cost.cuh
 *      definition of the common cost-computing function
 *
 * src/backend/optimizer/gpuqo/gpuqo_cost.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_COST_CUH
#define GPUQO_COST_CUH

#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"

#define BASEREL_COEFF   0.2
#define HASHJOIN_COEFF  1
#define INDEXSCAN_COEFF 2
#define SORT_COEFF      2

#define COST_FUNCTION_OVERHEAD 3000L

template<typename BitmapsetN>
__host__ __device__
static bool has_useful_index(BitmapsetN left_rel_id, BitmapsetN right_rel_id, 
                            GpuqoPlannerInfo<BitmapsetN>* info){
    if (right_rel_id.size() != 1)  // inner must be base rel
        return false;

    // -1 since it's 1-indexed, 
    // another -1 since relation with id 0b10 is at index 0 and so on
    int baserel_right_idx = right_rel_id.lowestPos() - 1;

    if (info->base_rels[baserel_right_idx].composite)
        return false;
    
    return left_rel_id.intersects(info->indexed_edge_table[baserel_right_idx]);
}

template<typename BitmapsetN>
__host__ __device__
static float baserel_cost(BaseRelation<BitmapsetN> &base_rel){
    return BASEREL_COEFF * base_rel.tuples;
}

template<typename BitmapsetN>
__host__ __device__
static float 
calc_join_cost(BitmapsetN left_rel_id, JoinRelation<BitmapsetN> &left_rel,
                BitmapsetN right_rel_id, JoinRelation<BitmapsetN> &right_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
{
    float hj_cost, nl_cost, inl_cost, sm_cost;
    float min_cost;

    // hash join
    hj_cost = HASHJOIN_COEFF * join_rel_rows + left_rel.cost + right_rel.cost;
    min_cost = hj_cost;

    // nested loop join
    nl_cost = left_rel.cost + left_rel.rows * right_rel.cost;
    min_cost = min(min_cost, nl_cost);
    
    // indexed nested loop join
    if (has_useful_index(left_rel_id, right_rel_id, info)){
        inl_cost = left_rel.cost 
            + INDEXSCAN_COEFF * left_rel.rows * max(
                                            join_rel_rows/left_rel.rows, 
                                            1.0);
        min_cost = min(min_cost, inl_cost);
    }

    // explicit sort merge
    sm_cost = left_rel.cost + right_rel.cost
        + SORT_COEFF * left_rel.rows * logf(left_rel.rows)
        + SORT_COEFF * right_rel.rows * logf(right_rel.rows);
    min_cost = min(min_cost, sm_cost);

#if defined(SIMULATE_COMPLEX_COST_FUNCTION) && COST_FUNCTION_OVERHEAD>0
    //Additional overhead to simulate complex cost functions
        volatile long counter = COST_FUNCTION_OVERHEAD;
        while (counter > 0){
            --counter;
        }
#endif

    return min_cost;
}

template<typename BitmapsetN>
__host__ __device__
static float fkselc_of_ec(int off_fk, BitmapsetN ec_relids, 
                    BitmapsetN outer_rels, BitmapsetN inner_rels, 
                    GpuqoPlannerInfo<BitmapsetN>* info) 
{
    while(!outer_rels.empty()){
        BitmapsetN out_id = outer_rels.lowest();
        int out_idx = (out_id.allLower() & ec_relids).size();
        BitmapsetN match = inner_rels & info->eq_class_fk[off_fk+out_idx];
        if (!match.empty()){
            int in_idx = match.lowestPos()-1;
            return 1.0 / max(1.0f, info->base_rels[in_idx].tuples);
        }

        outer_rels ^= out_id;
    }
    return NANF;
}

template<typename BitmapsetN>
__host__ __device__
static float 
estimate_ec_selectivity(BitmapsetN ec_relids, int off_sels, int off_fks,
                        BitmapsetN left_rel_id, BitmapsetN right_rel_id,
                        GpuqoPlannerInfo<BitmapsetN>* info)
{
    int size = ec_relids.size();
    BitmapsetN match_l = ec_relids & left_rel_id;
    BitmapsetN match_r = ec_relids & right_rel_id;

    if (match_l.empty() || match_r.empty())
        return 1.0f;

    // first check if any foreign key selectivity applies

    float fk_sel = NANF;
    fk_sel = fkselc_of_ec(off_fks, ec_relids, match_l, match_r, info);
    if (!isnan(fk_sel))
        return fk_sel;

    // try also swapping relations if nothing was found
    fk_sel = fkselc_of_ec(off_fks, ec_relids, match_r, match_l, info);
    if (!isnan(fk_sel)) // found fk selectivity -> apply it
        return fk_sel;

    // not found fk selectivity -> estimate from eq class

    // more than one on the same equivalence class may match
    // just take the lowest one (already done in allLower)

    int idx_l = (match_l.allLower() & ec_relids).size();
    int idx_r = (match_r.allLower() & ec_relids).size();
    int idx = eqClassIndex(idx_l, idx_r, size);

    return info->eq_class_sels[off_sels+idx];
}

template<typename BitmapsetN>
__host__ __device__
static float 
estimate_join_selectivity(BitmapsetN left_rel_id, BitmapsetN right_rel_id,
    GpuqoPlannerInfo<BitmapsetN>* info)
{
    float sel = 1.0;

    // for each ec that involves any baserel on the left and on the right,
    // get its selectivity.
    // NB: one equivalence class may only apply a selectivity once so the lowest
    // matching id on both sides is kept
    int off_sels = 0;
    int off_fks = 0;
    for (int i=0; i<info->n_eq_classes; i++){
        BitmapsetN ec_relids = info->eq_classes[i];
        
        sel *= estimate_ec_selectivity(
            info->eq_classes[i], off_sels, off_fks,
            left_rel_id, right_rel_id, info
        );
           
        off_sels += eqClassNSels(ec_relids.size());
        off_fks += ec_relids.size();
    }
    
    return sel;
}

template<typename BitmapsetN>
__host__ __device__
static float 
estimate_join_rows(BitmapsetN left_rel_id, JoinRelation<BitmapsetN> &left_rel,
                BitmapsetN right_rel_id, JoinRelation<BitmapsetN> &right_rel,
                GpuqoPlannerInfo<BitmapsetN>* info)
{
    float sel = estimate_join_selectivity(left_rel_id, right_rel_id, info);
    float rows = sel * left_rel.rows * right_rel.rows;

    // clamp the number of rows
    return rows > 1 ? round(rows) : 1;
}

#endif
