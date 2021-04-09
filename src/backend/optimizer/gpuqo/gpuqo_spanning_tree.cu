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

void minimumSpanningTree(GpuqoPlannerInfo *info){
    RelationID S = info->base_rels[0].id;
    RelationID out_relid = BMS32_EMPTY;

    EdgeMask out_edges[32];
    JoinRelationDetailed base_joinrels[32];

    for (int i=0; i < info->n_rels; i++){
        out_relid = BMS32_UNION(out_relid, info->base_rels[i].id);
        out_edges[i] = BMS32_EMPTY;

        JoinRelationDetailed t;
        t.id = info->base_rels[i].id;
        t.left_rel_id = 0; 
        t.right_rel_id = 0; 
        t.cost = baserel_cost(info->base_rels[i]); 
        t.rows = info->base_rels[i].rows; 
        base_joinrels[i] = t;
    }

    while (S != out_relid){
        float min = std::numeric_limits<float>::max();
        int argmin_in, argmin_out;
        for (int i=0; i < info->n_rels; i++){
            if (BMS32_INTERSECTS(S, BMS32_NTH(i+1))){
                RelationID edges = BMS32_DIFFERENCE(
                    info->edge_table[i],
                    S
                );
                for (int j=0; j < info->n_rels; j++){
                    if (BMS32_INTERSECTS(edges, BMS32_NTH(j+1))){
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

        S = BMS32_SET(S, argmin_out+1);
        out_edges[argmin_in] = BMS32_SET(out_edges[argmin_in], argmin_out+1);
        out_edges[argmin_out] = BMS32_SET(out_edges[argmin_out], argmin_in+1);
    }
    memcpy(info->edge_table, out_edges, info->n_rels * sizeof(EdgeMask));
}


static
RelationID buildSubTreesDFS(int idx, int parent_idx, 
                            RelationID* subtrees, EdgeMask* edge_table){
    RelationID subtree = BMS32_NTH(idx+1);
    RelationID N = edge_table[idx];
    while (N != BMS32_EMPTY){
        int child_idx = BMS32_LOWEST_POS(N)-2;
        if (child_idx != parent_idx){
            subtree = BMS32_UNION(subtree, 
                buildSubTreesDFS(child_idx, idx, subtrees, edge_table)
            );
        }
        N = BMS32_UNSET(N, child_idx+1);
    }

    subtrees[idx] = subtree;
    return subtree;
}

void buildSubTrees(RelationID* subtrees, GpuqoPlannerInfo *info){
    buildSubTreesDFS(0, -1, subtrees, info->edge_table);
}
