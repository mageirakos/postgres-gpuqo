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
    RelationID out_relid = RelationID(0);

    EdgeMask out_edges[RelationID::SIZE];
    JoinRelationDetailed base_joinrels[RelationID::SIZE];

    for (int i=0; i < info->n_rels; i++){
        out_relid |= info->base_rels[i].id;
        out_edges[i] = RelationID(0);

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
            if (S.isSet(i+1)){
                RelationID edges = info->edge_table[i] - S;
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
    memcpy(info->edge_table, out_edges, info->n_rels * sizeof(EdgeMask));
}


static
RelationID buildSubTreesDFS(int idx, int parent_idx, 
                            RelationID* subtrees, EdgeMask* edge_table){
    RelationID subtree = RelationID::nth(idx+1);
    RelationID N = edge_table[idx];
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

void buildSubTrees(RelationID* subtrees, GpuqoPlannerInfo *info){
    buildSubTreesDFS(0, -1, subtrees, info->edge_table);
}
