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

__host__ __device__
static bool has_useful_index(JoinRelation &left_rel, JoinRelation &right_rel,
                    GpuqoPlannerInfo* info){
    if (BMS32_SIZE(right_rel.id) != 1)  // inner must be base rel
        return false;
    // -1 since it's 1-indexed, 
    // another -1 since relation with id 0b10 is at index 0 and so on
    int baserel_right_idx = BMS32_LOWEST_POS(right_rel.id) - 2;
    
    return BMS32_INTERSECTS(
        left_rel.id, 
        info->indexed_edge_table[baserel_right_idx]
    );
}

__host__ __device__
static float baserel_cost(BaseRelation &base_rel){
    return BASEREL_COEFF * base_rel.tuples;
}

__host__ __device__
static float 
hash_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
               JoinRelation &right_rel, GpuqoPlannerInfo* info){
    return HASHJOIN_COEFF * join_rel.rows + left_rel.cost + right_rel.cost;
}

__host__ __device__
static float 
nl_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
               JoinRelation &right_rel, GpuqoPlannerInfo* info){
    return left_rel.cost + left_rel.rows * right_rel.cost;
}

__host__ __device__
static float 
inl_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
               JoinRelation &right_rel, GpuqoPlannerInfo* info){
    return left_rel.cost + INDEXSCAN_COEFF * left_rel.rows * max(join_rel.rows/left_rel.rows, 1.0);
}

__host__ __device__
static float 
sm_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
               JoinRelation &right_rel, GpuqoPlannerInfo* info){
    return left_rel.cost + right_rel.cost
        + SORT_COEFF * left_rel.rows * log(left_rel.rows)
        + SORT_COEFF * right_rel.rows * log(right_rel.rows);
}

__host__ __device__
static float 
calc_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, GpuqoPlannerInfo* info)
{
    float min_cost;

    // hash join
    min_cost = hash_join_cost(join_rel, left_rel, right_rel, info);

    // nested loop join
    min_cost = min(min_cost, nl_join_cost(join_rel, left_rel, right_rel, info));
    
    // indexed nested loop join
    if (has_useful_index(left_rel, right_rel, info)){
        min_cost = min(min_cost, inl_join_cost(join_rel, left_rel, right_rel, info));
    }

    // explicit sort merge
    min_cost = min(min_cost, sm_join_cost(join_rel, left_rel, right_rel, info));

    return min_cost;
}

__host__ __device__
static float 
estimate_join_selectivity(JoinRelation &left_rel, JoinRelation &right_rel, 
                            GpuqoPlannerInfo* info) 
{
    float sel = 1.0;

    // check fk with base relations
    if (BMS32_SIZE(left_rel.id) == 1 && BMS32_SIZE(right_rel.id) == 1){
        int left_rel_idx = BMS32_LOWEST_POS(left_rel.id)-2;
        int right_rel_idx = BMS32_LOWEST_POS(right_rel.id)-2;
        float fksel = info->fk_selecs[left_rel_idx * info->n_rels + right_rel_idx];

        if(!isnan(fksel)){
            return fksel;
        }
    }
    
    // for each ec that involves any baserel on the left and on the right,
    // get its selectivity.
    // NB: one equivalence class may only apply a selectivity once so the lowest
    // matching id on both sides is kept
    EqClassInfo* ec = info->eq_classes;
    while (ec != NULL){
        RelationID match_l = BMS32_INTERSECTION(ec->relids, left_rel.id);
        RelationID match_r = BMS32_INTERSECTION(ec->relids, right_rel.id);

        if (match_l != BMS32_EMPTY && match_r != BMS32_EMPTY){
            // more than one on the same equivalence class may match
            // just take the lowest one (already done in BMS32_SET_ALL_LOWER)

            int idx_l = BMS32_SIZE(
                BMS32_INTERSECTION(
                    BMS32_SET_ALL_LOWER(match_l),
                    ec->relids
                )
            );
            int idx_r = BMS32_SIZE(
                BMS32_INTERSECTION(
                    BMS32_SET_ALL_LOWER(match_r),
                    ec->relids
                )
            );
            int size = BMS32_SIZE(ec->relids);

            sel *= ec->sels[idx_l*size+idx_r];
        }
        ec = ec->next;
    }
    
    return sel;
}

__host__ __device__
static float 
estimate_join_rows(JoinRelation &left_rel, JoinRelation &right_rel, GpuqoPlannerInfo* info) 
{
    float sel = estimate_join_selectivity(left_rel, right_rel, info);
    float rows = sel * left_rel.rows * right_rel.rows;

    // clamp the number of rows
    return rows > 1 ? round(rows) : 1;
}

__host__ __device__
static void 
_compute_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, GpuqoPlannerInfo* info)
{    
    join_rel.rows = estimate_join_rows(left_rel, right_rel, info);
    join_rel.cost = calc_join_cost(join_rel, left_rel, right_rel, info);
}

__host__ __device__
static void 
compute_join_cost(JoinRelation &join_rel, JoinRelation &left_rel,
                    JoinRelation &right_rel, GpuqoPlannerInfo* info)
{    
    _compute_join_cost(join_rel, left_rel, right_rel, info);
}

__device__
static void 
make_join_rel(JoinRelation &join_rel, uint32_t left_idx, JoinRelation &left_rel,
                uint32_t right_idx, JoinRelation &right_rel, 
                GpuqoPlannerInfo* info)
{    
    join_rel.id = BMS32_UNION(left_rel.id, right_rel.id);
    join_rel.left_relation_id = left_rel.id;
    join_rel.left_relation_idx = left_idx;
    join_rel.right_relation_id = right_rel.id;
    join_rel.right_relation_idx = right_idx;
    join_rel.edges = BMS32_UNION(left_rel.edges, right_rel.edges);
    _compute_join_cost(join_rel, left_rel, right_rel, info);
}

__device__
static void 
make_join_rel(JoinRelation &join_rel, JoinRelation &left_rel,
              JoinRelation &right_rel, GpuqoPlannerInfo* info)
{    
    join_rel.id = BMS32_UNION(left_rel.id, right_rel.id);
    join_rel.left_relation_id = left_rel.id;
    join_rel.left_relation_idx = left_rel.id;
    join_rel.right_relation_id = right_rel.id;
    join_rel.right_relation_idx = right_rel.id;
    join_rel.edges = BMS32_UNION(left_rel.edges, right_rel.edges);
    _compute_join_cost(join_rel, left_rel, right_rel, info);
}


#endif
