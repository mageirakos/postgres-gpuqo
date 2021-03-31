/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_bicc.cuh
 *      device functions for BiCC decomposition
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_tree.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_DPSUB_ENUM_BICC_CUH
#define GPUQO_DPSUB_ENUM_BICC_CUH

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_dpsub.cuh"


struct bfs_bicc_bfs_lv_ret{
    uint32_t l;
    uint32_t vid_low;
    RelationID V_u;
};

__device__ 
static bfs_bicc_bfs_lv_ret bfs_bicc_bfs_lv(const uint32_t* L, 
                                        uint32_t v, uint32_t u,
                                        RelationID invalid,
                                        RelationID relid, const EdgeMask* edges)
{
    RelationID visited = BMS32_UNION(invalid, BMS32_CMP(relid));
    RelationID Q = BMS32_NTH(u);
    RelationID V_u = BMS32_NTH(u);

    visited = BMS32_SET(visited, u);
    visited = BMS32_SET(visited, v);

    uint32_t vid_low = u;

    while (Q != BMS32_EMPTY){
        int x = BMS32_LOWEST_POS(Q)-1;

        EdgeMask e = BMS32_DIFFERENCE(edges[x-1], visited);

        while (e != BMS32_EMPTY){
            int w = BMS32_LOWEST_POS(e)-1;

            if (L[w] < L[u]){
                return (bfs_bicc_bfs_lv_ret){
                    .l = L[w],
                    .vid_low = 0,
                    .V_u = BMS32_EMPTY
                };
            } else {
                Q = BMS32_SET(Q, w);
                V_u = BMS32_SET(V_u, w);
                visited = BMS32_SET(visited, w);
                if (w < vid_low)
                    vid_low = w;
            }

            e = BMS32_UNSET(e, w);
        }

        Q = BMS32_UNSET(Q, x);
    }


    return (bfs_bicc_bfs_lv_ret){
        .l = L[u],
        .vid_low = vid_low,
        .V_u = V_u
    };
}

__device__ 
static void bfs_bicc_bfs(RelationID relid, const EdgeMask* edges,
                            volatile uint32_t* P, volatile uint32_t* L, 
                            volatile RelationID* LQ)
{
    __shared__ RelationID shared_visited[BLOCK_DIM/WARP_SIZE];

    int t_off = threadIdx.x >> 5;

    volatile RelationID* visited = shared_visited + t_off;
    
    if (LANE_ID == 0){
        uint32_t r = BMS32_LOWEST_POS(relid)-1;
        P[r] = r;
        L[r] = 0;
        LQ[0] = BMS32_NTH(r);
        *visited = BMS32_UNION(BMS32_NTH(r), BMS32_CMP(relid));
    }

    __syncwarp();

    for (int i=0; i<31; i++){
        if (BMS32_INTERSECTS(BMS32_NTH(LANE_ID), LQ[i])){
            int x = LANE_ID;

            EdgeMask e = BMS32_DIFFERENCE(edges[x-1], *visited);
            __syncwarp();
            atomicOr((unsigned*)&LQ[i+1], (unsigned)e);
            atomicOr((unsigned*)visited, (unsigned)e);

            LOG_DEBUG("bfs_bicc_bfs(%u): i=%d, x=%d, e=%u\n", 
                        relid, i, x, e);

            while (e != BMS32_EMPTY){
                int w = BMS32_LOWEST_POS(e)-1;

                L[w] = i+1; // same for all threads, concurrency not important
                P[w] = x;   // different but not important which one wins

                e = BMS32_UNSET(e, w);
            }
        }
        __syncwarp();
    }
    
}

__device__
static int bfs_bicc(RelationID relid, const EdgeMask* edges, RelationID *blocks)
{
    __shared__ uint32_t shared_P[32*BLOCK_DIM/32];
    __shared__ uint32_t shared_L[32*BLOCK_DIM/32];
    __shared__ uint32_t shared_Low[32*BLOCK_DIM/32];
    __shared__ uint32_t shared_Par[32*BLOCK_DIM/32];
    __shared__ RelationID shared_LQ[32*BLOCK_DIM/32];
    __shared__ RelationID shared_invalid[BLOCK_DIM/32];

    int t_off = threadIdx.x >> 5;

    uint32_t *P = shared_P + t_off*32;
    uint32_t *L = shared_L + t_off*32;
    uint32_t *Low = shared_Low + t_off*32;
    uint32_t *Par = shared_Par + t_off*32;
    RelationID *LQ = shared_LQ + t_off*32;
    RelationID *invalid = shared_invalid + t_off;

    P[LANE_ID] = 0;
    L[LANE_ID] = 0;
    LQ[LANE_ID] = BMS32_EMPTY;
    Low[LANE_ID] = LANE_ID;
    Par[LANE_ID] = LANE_ID;

    if (LANE_ID == 0)
        *invalid = BMS32_EMPTY;

    int n_blocks = 0;

    bfs_bicc_bfs(relid, edges, P, L, LQ);

    LOG_DEBUG("L[%d]=%u\tP[%d]=%u\n", LANE_ID, L[LANE_ID], LANE_ID, P[LANE_ID]);

    for (int i = 31; i>0; i--){
        bfs_bicc_bfs_lv_ret s;
        uint32_t u = LANE_ID;
        uint32_t v = P[u];
        bool p1 = BMS32_INTERSECTS(BMS32_NTH(LANE_ID), LQ[i]) && Par[u] == u;
        bool p2 = false;

        if (p1){
            LOG_DEBUG("bfs_bicc(%u): iter %d, node %u (parent: %u)\n",
                                                             relid, i, u, v);

            s = bfs_bicc_bfs_lv(L, v, u, *invalid, relid, edges);

            LOG_DEBUG("bfs_bicc_bfs_lv(%u, %u, %u): "
                          "l=%u, vid_low=%u, V_u=%u\n",
                          relid, v, u, s.l, s.vid_low, s.V_u);

            p2 = (s.l >= L[u]
                && BMS32_LOWEST_POS(
                        BMS32_INTERSECTION(s.V_u, LQ[i])
                    )-1 == u
            );
                        
        }
        
        unsigned pthBlt = __ballot_sync(WARP_MASK, p2);
        int wScan = __popc(pthBlt & LANE_MASK_LE);

        if (p2){
            int idx = n_blocks+wScan-1;
            blocks[idx] = BMS32_SET(s.V_u, v);
            LOG_DEBUG("[%u] %u is articulation (block[%d]: %u)\n", 
                        relid, v, idx, blocks[idx]);
            while (s.V_u != BMS32_EMPTY){
                int w = BMS32_LOWEST_POS(s.V_u)-1;
                Low[w] = s.vid_low;
                Par[w] = v;
                atomicOr(invalid, BMS32_NTH(w));

                s.V_u = BMS32_UNSET(s.V_u, w);
            }
        }
        int new_blocks = __popc(pthBlt); 
        n_blocks += new_blocks;
        __syncwarp();
    }

    return n_blocks;
}

struct dpsubEnumerateBiCC : public pairs_enum_func_t 
{
    HashTable32bit memo;
    GpuqoPlannerInfo* info;
    int n_splits;
public:
    dpsubEnumerateBiCC(
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
    
        Assert(n_splits == 32);

        int t_off = threadIdx.x >> 5;

        int n_active = __popc(__activemask());
        __shared__ EdgeMask edge_table[32];
        for (int i = threadIdx.x; i < info->n_rels; i+=n_active){
            edge_table[i] = info->edge_table[i];
        }
        __syncthreads();

        __shared__ RelationID shared_blocks[32*BLOCK_DIM/32];
        RelationID* blocks = shared_blocks + t_off*32;

        int n_blocks = bfs_bicc(relid, edge_table, blocks);

        LOG_DEBUG("%u has %d blocks\n", relid, n_blocks);

        uint32_t n_possible_joins = 0;
        for (int i=0; i<n_blocks; i++){
            n_possible_joins += (1 << BMS32_SIZE(blocks[i])) - 2;
        }

        LOG_DEBUG("relid=%u: n_possible_joins=%u\n", relid, n_possible_joins);

        uint32_t n_joins = ceil_div(n_possible_joins, n_splits);
        uint32_t from_i = cid*n_joins;
        uint32_t i_block = 0;
        uint32_t psum = (1 << BMS32_SIZE(blocks[i_block])) - 2;
        uint32_t prev_psum = 0;
    
        for (uint32_t off_i = 0; 
            off_i < n_joins && (from_i+off_i) < n_possible_joins; 
            off_i++
        ){
            uint32_t i = from_i + off_i;
            while (i >= psum){
                prev_psum = psum;
                psum += (1 << BMS32_SIZE(blocks[++i_block])) - 2;
            }
            
            RelationID id = i - prev_psum + 1;
            RelationID block_left_id = BMS32_EXPAND_TO_MASK(id,blocks[i_block]);
            RelationID block_right_id = BMS32_DIFFERENCE(blocks[i_block], block_left_id);
            RelationID permitted = BMS32_DIFFERENCE(relid, block_right_id);

            LOG_DEBUG("relid=%u, i=%u: id=%u, block[%d]=%u, bl=%u, br=%u, p=%u\n",
                        relid, i, id, i_block, blocks[i_block], 
                        block_left_id, block_right_id, permitted);

            
            if (is_connected(block_left_id, edge_table)
                && is_connected(block_right_id, edge_table)
            ){
                RelationID l = grow(block_left_id, permitted, 
                                edge_table);
                RelationID r = BMS32_DIFFERENCE(relid, l);
                
                Assert(l != BMS32_EMPTY && r != BMS32_EMPTY);

                LOG_DEBUG("[%3d,%3d]: %u %u (%u)\n", 
                    blockIdx.x,
                    threadIdx.x,
                    l,
                    r, 
                    relid
                );

                JoinRelation left_rel = *memo.lookup(l);
                JoinRelation right_rel = *memo.lookup(r);
    
                do_join(jr_out, l, left_rel, r, right_rel, info);
            }
        }
    
        return jr_out;
    }
};

#endif              // GPUQO_DPSUB_ENUM_BICC_CUH
