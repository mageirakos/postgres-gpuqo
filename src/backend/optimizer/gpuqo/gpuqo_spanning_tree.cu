/*------------------------------------------------------------------------
 *
 * gpuqo_spanning_tree.c
 *	  procedure to extract minimum spanning tree of graph
 *
 * src/backend/optimizer/gpuqo/gpuqo_spanning_tree.c
 *
 *-------------------------------------------------------------------------
 */

#include <limits>
#include "gpuqo.cuh"
#include "gpuqo_cost.cuh"

bool gpuqo_spanning_tree_enable;

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN> *
minimumSpanningTree(GpuqoPlannerInfo<BitmapsetN> *info)
{
    BitmapsetN S = info->base_rels[0].id;
    BitmapsetN out_relid = BitmapsetN(0);

    BitmapsetN *out_edges = new BitmapsetN[info->n_rels];
    JoinRelationDetailed<BitmapsetN> *base_joinrels = new JoinRelationDetailed<BitmapsetN>[info->n_rels];

    for (int i=0; i < info->n_rels; i++){
        out_relid |= info->base_rels[i].id;
        out_edges[i] = BitmapsetN(0);

        JoinRelationDetailed<BitmapsetN> t;
        t.id = info->base_rels[i].id;
        t.left_rel_id = 0; 
        t.right_rel_id = 0; 
        t.cost = cost_baserel(info->base_rels[i]); 
        t.width = info->base_rels[i].width; 
        t.rows = info->base_rels[i].rows; 
        base_joinrels[i] = t;
    }

    while (S != out_relid){
        float min = std::numeric_limits<float>::max();
        int argmin_in, argmin_out;
        for (int i=0; i < info->n_rels; i++){
            if (S.isSet(i+1)){
                BitmapsetN edges = info->edge_table[i] - S;
                for (int j=0; j < info->n_rels; j++){
                    if (edges.isSet(j+1)){
                        float sel = estimate_join_rows(
                            base_joinrels[i].id,
                            base_joinrels[i],
                            base_joinrels[j].id,
                            base_joinrels[j],
                            info
                        );
                        if (sel < min){
                            min = sel;
                            argmin_in = i;
                            argmin_out = j;
                        }
                    }
                }
            }
        }

        S.set(argmin_out+1);
        out_edges[argmin_in].set(argmin_out+1);
        out_edges[argmin_out].set(argmin_in+1);
    }
    GpuqoPlannerInfo<BitmapsetN> *clone_info = cloneGpuqoPlannerInfo(info);
    std::copy(out_edges, out_edges + info->n_rels, clone_info->edge_table);
    delete[] out_edges;
    delete[] base_joinrels;

    return clone_info;
}

template GpuqoPlannerInfo<Bitmapset32> *minimumSpanningTree<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32> *info);
template GpuqoPlannerInfo<Bitmapset64> *minimumSpanningTree<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64> *info);
template GpuqoPlannerInfo<BitmapsetDynamic> *minimumSpanningTree<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic> *info);


template<typename BitmapsetN>
static
BitmapsetN buildSubTreesDFS(int idx, int parent_idx, 
                            BitmapsetN* subtrees, BitmapsetN* edge_table){
    BitmapsetN subtree = BitmapsetN::nth(idx+1);
    BitmapsetN N = edge_table[idx];
    while (!N.empty()){
        int child_idx = N.lowestPos()-1;
        if (child_idx != parent_idx){
            subtree |= buildSubTreesDFS(child_idx, idx, subtrees, edge_table);
        }
        N.unset(child_idx+1);
    }

    subtrees[idx] = subtree;
    return subtree;
}

template<typename BitmapsetN>
void buildSubTrees(BitmapsetN* subtrees, GpuqoPlannerInfo<BitmapsetN> *info){
    buildSubTreesDFS(0, -1, subtrees, info->edge_table);
}

template void buildSubTrees<Bitmapset32>(Bitmapset32* subtrees, GpuqoPlannerInfo<Bitmapset32> *info);
template void buildSubTrees<Bitmapset64>(Bitmapset64* subtrees, GpuqoPlannerInfo<Bitmapset64> *info);
template void buildSubTrees<BitmapsetDynamic>(BitmapsetDynamic* subtrees, GpuqoPlannerInfo<BitmapsetDynamic> *info);
