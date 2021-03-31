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
    JoinRelation operator()(RelationID relid, uint32_t cid)
    {
        JoinRelation jr_out;
        jr_out.cost = INFD;

        uint32_t n_possible_joins = BMS32_SIZE(relid);

        int n_active = __popc(__activemask());
        __shared__ EdgeMask edge_table[32];
        for (int i = threadIdx.x; i < info->n_rels; i+=n_active){
            edge_table[i] = info->edge_table[i];
        }
        __syncthreads();

        for (uint32_t i = cid; i < n_possible_joins; i += n_splits){
            RelationID base_rel_id = BMS32_EXPAND_TO_MASK(BMS32_NTH(i), relid);
            RelationID permitted = BMS32_DIFFERENCE(
                relid, 
                // TODO check if I can rely on this in general...
                BMS32_SET_ALL_LOWER(base_rel_id)
            );

            LOG_DEBUG("%d %d: %u (%u)\n", 
                blockIdx.x,
                threadIdx.x,
                base_rel_id,
                relid
            );

            RelationID l = grow(base_rel_id, permitted, edge_table);
            RelationID r = BMS32_DIFFERENCE(relid, l);

            if (l != BMS32_EMPTY && r != BMS32_EMPTY){
                JoinRelation left_rel = *memo.lookup(l);
                JoinRelation right_rel = *memo.lookup(r);

                LOG_DEBUG("%d %d: %u %u (%u)\n", 
                    blockIdx.x,
                    threadIdx.x,
                    l,
                    r, 
                    relid
                );

                do_join(jr_out, l, left_rel, r, right_rel, info);
                do_join(jr_out, r, right_rel, l, left_rel, info);
            }
        }

        return jr_out;
    }
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
    JoinRelation operator()(RelationID relid, uint32_t cid)
    { 
        JoinRelation jr_out;
        jr_out.cost = INFD;
    
        uint32_t n_possible_joins = BMS32_SIZE(relid);
    
        for (uint32_t i = cid; i < n_possible_joins; i += n_splits){
            RelationID base_rel_id = BMS32_EXPAND_TO_MASK(BMS32_NTH(i), relid);
            int base_rel_idx = BMS32_LOWEST_POS(base_rel_id)-2;
    
            Assert(base_rel_idx < info->n_rels);
    
            RelationID S = BMS32_INTERSECTION(
                info->subtrees[base_rel_idx],
                relid
            );
    
            RelationID l = S;
            RelationID r = BMS32_DIFFERENCE(relid, S);
    
            if (l != BMS32_EMPTY && r != BMS32_EMPTY){
                JoinRelation left_rel = *memo.lookup(l);
                JoinRelation right_rel = *memo.lookup(r);
    
                LOG_DEBUG("%d %d: %u %u (%u)\n", 
                    blockIdx.x,
                    threadIdx.x,
                    l,
                    r, 
                    relid
                );
    
                do_join(jr_out, l, left_rel, r, right_rel, info);
                do_join(jr_out, r, right_rel, l, left_rel, info);
            }
        }
    
        return jr_out;
    }
};

#endif              // GPUQO_DPSUB_ENUM_TREE_CUH