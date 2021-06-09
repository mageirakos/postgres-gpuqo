/*------------------------------------------------------------------------
 *
 * gpuqo_row_estimation.cuh
 *      definition of the common cost-computing function
 *
 * src/backend/optimizer/gpuqo/gpuqo_row_estimation.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_ROW_ESTIMATION_CUH
#define GPUQO_ROW_ESTIMATION_CUH

#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"

template<typename BitmapsetN>
__host__ __device__
static float fkselc_of_ec(int off_fk, BitmapsetN ec_relids, 
                    BitmapsetN outer_rels, BitmapsetN inner_rels, 
                    GpuqoPlannerInfo<BitmapsetN>* info) 
{
    while(!outer_rels.empty()){
        BitmapsetN out_id = outer_rels.lowest();
        int out_idx = (out_id.allLower() & ec_relids).size();
        BitmapsetN match = inner_rels & info->eq_classes.fks[off_fk+out_idx];
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

    return info->eq_classes.sels[off_sels+idx];
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
    for (int i=0; i<info->eq_classes.n; i++){
        BitmapsetN ec_relids = info->eq_classes.relids[i];
        
        sel *= estimate_ec_selectivity(
            info->eq_classes.relids[i], off_sels, off_fks,
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

template<typename BitmapsetN>
__host__ __device__
static float 
estimate_join_rows(QueryTree<BitmapsetN> &left_rel, 
                   QueryTree<BitmapsetN> &right_rel, 
                   GpuqoPlannerInfo<BitmapsetN>* info)
{
    float sel = estimate_join_selectivity(left_rel.id, right_rel.id, info);
    float rows = sel * left_rel.rows * right_rel.rows;

    // clamp the number of rows
    return rows > 1 ? round(rows) : 1;
}


template<typename BitmapsetN>
__host__ __device__
static int 
get_join_width(BitmapsetN left_rel_id, JoinRelation<BitmapsetN> &left_rel,
                BitmapsetN right_rel_id, JoinRelation<BitmapsetN> &right_rel,
                GpuqoPlannerInfo<BitmapsetN>* info) {
    // TODO
    return left_rel.width + right_rel.width;
}

#endif // GPUQO_ROW_ESTIMATION_CUH
