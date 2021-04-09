/*------------------------------------------------------------------------
 *
 * gpuqo_bfs_indexing.cu
 *	  utilities to remap indices so that they are BFS-coherent
 *
 * src/backend/optimizer/gpuqo/gpuqo_bfs_indexing.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo.cuh"

void makeBFSIndexRemapTables(int *remap_table_fw, int *remap_table_bw, GpuqoPlannerInfo* info){
    int bfs_queue[32];
    int bfs_queue_front_idx = 0;
    int bfs_queue_back_idx = 0;

    int bfs_idx = 0;
    
    bfs_queue[bfs_queue_back_idx++] = 0;

    Bitmapset32 seen = BMS32_NTH(1);
    while (bfs_queue_front_idx != bfs_queue_back_idx && bfs_idx < info->n_rels){
        int base_rel_idx = bfs_queue[bfs_queue_front_idx++];
        
        BaseRelation* r = &info->base_rels[base_rel_idx];
        EdgeMask edges = info->edge_table[base_rel_idx];

        remap_table_fw[base_rel_idx] = bfs_idx;
        remap_table_bw[bfs_idx] = base_rel_idx;
        bfs_idx++;

        while (edges != BMS32_EMPTY){
            RelationID next_r = BMS32_LOWEST(edges);
            int next = BMS32_LOWEST_POS(edges) - 2;

            if (!BMS32_INTERSECTS(seen, next_r)){
                bfs_queue[bfs_queue_back_idx++] = next;
            }
            
            edges = BMS32_DIFFERENCE(edges, BMS32_LOWEST(edges));
        }
        seen = BMS32_UNION(seen, info->edge_table[base_rel_idx]);
    }
}

RelationID remapRelid(RelationID id, int *remap_table){
    RelationID in = id;
    RelationID out = BMS32_EMPTY;
    while (id != BMS32_EMPTY){
        int pos = BMS32_LOWEST_POS(id)-2;
        out = BMS32_UNION(out, BMS32_NTH(remap_table[pos]+1));
        id = BMS32_DIFFERENCE(id, BMS32_NTH(pos+1));
    }

    return out;
}

void remapEdgeTable(EdgeMask* edge_table, int n, int* remap_table){
    EdgeMask* edge_table_tmp = (EdgeMask*) malloc(n*sizeof(EdgeMask));
    memcpy(edge_table_tmp, edge_table, n*sizeof(EdgeMask));
    for (int i = 0; i < n; i++){
        edge_table[remap_table[i]] = remapRelid(edge_table_tmp[i], remap_table);
    }

    free(edge_table_tmp);
}

void remapPlannerInfo(GpuqoPlannerInfo* info, int* remap_table){
    for (int i = 0; i < info->n_rels; i++){
        info->base_rels[i].id = remapRelid(info->base_rels[i].id, remap_table);
    }
    remapEdgeTable(info->edge_table, info->n_rels, remap_table);
    remapEdgeTable(info->indexed_edge_table, info->n_rels, remap_table);

    // even though it's not an edge_table, it has the same format
    if (gpuqo_spanning_tree_enable)
        remapEdgeTable(info->subtrees, info->n_rels, remap_table);

    for (int i=0; i<info->n_eq_classes; i++){
        info->eq_classes[i] = remapRelid(info->eq_classes[i], remap_table);
    }

    for (int i=0; i<info->n_fk_selecs; i++){
        info->fk_selec_idxs[i] = remap_table[info->fk_selec_idxs[i]];
    }
}

void remapQueryTree(QueryTree* qt, int* remap_table){
    if (qt == NULL)
        return;

    qt->id = remapRelid(qt->id, remap_table);

    remapQueryTree(qt->left, remap_table);
    remapQueryTree(qt->right, remap_table);
}
