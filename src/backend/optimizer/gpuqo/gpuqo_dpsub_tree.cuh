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

template<typename BitmapsetN>
uint32_t dpsub_tree_evaluation(int iter, uint64_t n_remaining_sets, 
                           uint64_t offset, uint32_t n_pending_sets, 
                           dpsub_iter_param_t<BitmapsetN> &params);

template<typename BitmapsetN>
struct dpsubEnumerateTreeSimple{
    __device__
    JoinRelation<BitmapsetN> operator()(BitmapsetN relid, 
                            uint32_t cid, int n_splits,
                            HashTableDpsub<BitmapsetN> &memo, 
                            GpuqoPlannerInfo<BitmapsetN>* info)
    {
        JoinRelation<BitmapsetN> jr_out;
        jr_out.cost.total = INFF;
        jr_out.cost.startup = INFF;

        uint32_t n_possible_joins = relid.size();

        for (uint32_t i = cid; i < n_possible_joins; i += n_splits){
            BitmapsetN base_rel_id = expandToMask(BitmapsetN::nth(i), relid);
            // TODO check if I can rely on this in general...
            BitmapsetN permitted = relid - base_rel_id.allLower();

            LOG_DEBUG("%d %d: %u (%u)\n", 
                blockIdx.x,
                threadIdx.x,
                base_rel_id.toUint(),
                relid.toUint()
            );

            BitmapsetN l = grow(base_rel_id, permitted, info->edge_table);
            BitmapsetN r = relid - l;

            if (!l.empty() && !r.empty()){
                Assert(is_connected(l, info->edge_table));
                Assert(is_connected(r, info->edge_table));
                Assert(is_disjoint(l, r));
                
                LOG_DEBUG("%d %d: %u %u (%u)\n", 
                    blockIdx.x,
                    threadIdx.x,
                    l.toUint(),
                    r.toUint(), 
                    relid.toUint()
                );

                JoinRelation<BitmapsetN> left_rel = *memo.lookup(l);
                JoinRelation<BitmapsetN> right_rel = *memo.lookup(r);
                do_join(jr_out, l, left_rel, r, right_rel, info);
                do_join(jr_out, r, right_rel, l, left_rel, info);
            }
        }

        return jr_out;
    }
};


template<typename BitmapsetN>
struct dpsubEnumerateTreeWithSubtrees{
    __device__
    JoinRelation<BitmapsetN> operator()(
                            BitmapsetN relid, 
                            uint32_t cid, int n_splits,
                            HashTableDpsub<BitmapsetN> &memo, 
                            GpuqoPlannerInfo<BitmapsetN>* info)
    { 
        JoinRelation<BitmapsetN> jr_out;
        jr_out.cost.total = INFF;
        jr_out.cost.startup = INFF;

        uint32_t n_possible_joins = relid.size();

        for (uint32_t i = cid; i < n_possible_joins; i += n_splits){
            BitmapsetN base_rel_id = expandToMask(BitmapsetN::nth(i), relid);
            int base_rel_idx = base_rel_id.lowestPos()-1;

            Assert(base_rel_idx < info->n_rels);

            BitmapsetN S = info->subtrees[base_rel_idx] & relid;

            BitmapsetN l = S;
            BitmapsetN r = relid - S;

            LOG_DEBUG("%d %d: %u %u (%u)\n", 
                blockIdx.x,
                threadIdx.x,
                l.toUint(),
                r.toUint(), 
                relid.toUint()
            );

            if (!l.empty() && !r.empty()){
                Assert(is_connected(l, info->edge_table));
                Assert(is_connected(r, info->edge_table));
                Assert(is_disjoint(l, r));

                JoinRelation<BitmapsetN> left_rel = *memo.lookup(l);
                JoinRelation<BitmapsetN> right_rel = *memo.lookup(r);

                do_join(jr_out, l, left_rel, r, right_rel, info);
                do_join(jr_out, r, right_rel, l, left_rel, info);
            }
        }

        return jr_out;
    }
};

#endif              // GPUQO_DPSUB_ENUM_TREE_CUH