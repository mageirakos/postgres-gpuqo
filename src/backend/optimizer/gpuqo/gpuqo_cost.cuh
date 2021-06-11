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
#include "gpuqo_row_estimation.cuh"

#ifdef GPUQO_COST_FUNCTION_COUT
#include "gpuqo_cost_cout.cuh"
#else
#include "gpuqo_cost_postgres.cuh"
#endif

#define COST_FUNCTION_OVERHEAD 3000L

template<typename BitmapsetN>
__host__ __device__
static bool 
is_inner_unique(BitmapsetN outer_rel_id, 
                BitmapsetN inner_rel_id, GpuqoPlannerInfo<BitmapsetN>* info)
{
    if (inner_rel_id.size() != 1)
        return false;

    for (int i=0, off_vars=0; 
        i<info->eq_classes.n; 
        off_vars += info->eq_classes.relids[i].size(),
        i++)
    {
        BitmapsetN ec_relids = info->eq_classes.relids[i];
        
        BitmapsetN match_l = ec_relids & outer_rel_id;
        BitmapsetN match_r = ec_relids & inner_rel_id;

        if (match_l.empty() || match_r.empty())
            continue;

        BitmapsetN in_id = match_r.lowest();
        int in_idx = (in_id.allLower() & ec_relids).size();

        VarInfo var = info->eq_classes.vars[off_vars+in_idx];

        if (var.index.available && var.index.unique)
            return true;
    }
    return false;
}

template<typename BitmapsetN>
__host__ __device__
static struct PathCost 
calc_join_cost(BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
{
    PathCost min_cost, tmp_cost;
    CostExtra extra;

    // extra.inner_unique = false;
    extra.inner_unique = is_inner_unique(outer_rel_id, inner_rel_id, info);
    extra.indexed_join_quals = false;
    extra.joinrows = join_rel_rows;

    min_cost.total = NANF;

    tmp_cost = cost_nestloop(outer_rel_id, outer_rel, inner_rel_id, inner_rel, extra, info);

    if (isnan(min_cost.total) || tmp_cost.total < min_cost.total)
        min_cost = tmp_cost;

    if (info->params.enable_hashjoin) {
        tmp_cost = cost_hashjoin(outer_rel_id, outer_rel, inner_rel_id, inner_rel, extra, info);

        if (tmp_cost.total < min_cost.total)
        min_cost = tmp_cost;
    }

    // Index nested loop join
    extra.indexed_join_quals = true;

    // if it is a base relation, check if I can use an index
    if (inner_rel_id.size() == 1) {
        BaseRelation<BitmapsetN>& baserel = info->base_rels[inner_rel_id.lowestPos()-1];
        if (!baserel.composite) {
            for (int i=0, off_vars=0; 
                i<info->eq_classes.n; 
                off_vars += info->eq_classes.relids[i].size(),
                i++)
            {
                BitmapsetN ec_relids = info->eq_classes.relids[i];
                
                BitmapsetN match_l = ec_relids & outer_rel_id;
                BitmapsetN match_r = ec_relids & inner_rel_id;

                if (match_l.empty() || match_r.empty())
                    continue;

                BitmapsetN in_id = match_r.lowest();
                int in_idx = (in_id.allLower() & ec_relids).size();

                VarInfo var = info->eq_classes.vars[off_vars+in_idx];

                if (!var.index.available)
                    continue;

                struct JoinRelation<BitmapsetN> indexed_inner_rel;
                indexed_inner_rel.left_rel_id = inner_rel.left_rel_id;
                indexed_inner_rel.right_rel_id = inner_rel.right_rel_id;
                indexed_inner_rel.rows = var.index.rows;
                indexed_inner_rel.cost = var.index.cost;

                tmp_cost = cost_nestloop(outer_rel_id, outer_rel, inner_rel_id, indexed_inner_rel, extra, info);
                if (tmp_cost.total < min_cost.total)
                    min_cost = tmp_cost;
            }
        }
    }

#if defined(SIMULATE_COMPLEX_COST_FUNCTION) && COST_FUNCTION_OVERHEAD>0
    //Additional overhead to simulate complex cost functions
        volatile long counter = COST_FUNCTION_OVERHEAD;
        while (counter > 0){
            --counter;
        }
#endif

    return min_cost;
}

#endif
