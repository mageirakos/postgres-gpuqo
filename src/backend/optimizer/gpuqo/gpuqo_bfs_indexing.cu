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

    RelationID seen = RelationID::nth(1);
    while (bfs_queue_front_idx != bfs_queue_back_idx && bfs_idx < info->n_rels){
        int base_rel_idx = bfs_queue[bfs_queue_front_idx++];
        
        EdgeMask edges = info->edge_table[base_rel_idx];

        remap_table_fw[base_rel_idx] = bfs_idx;
        remap_table_bw[bfs_idx] = base_rel_idx;
        bfs_idx++;

        while (!edges.empty()){
            RelationID next_r = edges.lowest();
            int next = edges.lowestPos() - 1;

            if (!seen.intersects(next_r)){
                bfs_queue[bfs_queue_back_idx++] = next;
            }
            
            edges -= edges.lowest();
        }
        seen |= info->edge_table[base_rel_idx];
    }
}

RelationID remapRelid(RelationID id, int *remap_table){
    RelationID in = id;
    RelationID out = RelationID(0);
    while (!id.empty()){
        int pos = id.lowestPos();
        out.set(remap_table[pos-1]+1);
        id.unset(pos);
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

void remapBaseRels(BaseRelation* base_rels, int n, int* remap_table){
    BaseRelation* base_rels_tmp = (BaseRelation*) malloc(n*sizeof(BaseRelation));
    memcpy(base_rels_tmp, base_rels, n*sizeof(BaseRelation));
    for (int i = 0; i < n; i++){
        base_rels[remap_table[i]] = base_rels_tmp[i];
        base_rels[remap_table[i]].id = remapRelid(base_rels_tmp[i].id, remap_table);
    }

    free(base_rels_tmp);
}

void remapEqClass(RelationID* eq_class, float* sels, int* remap_table){
    RelationID new_eq_class = remapRelid(*eq_class, remap_table);
    int s = eq_class->size();
    int n = eqClassNSels(s);

    float* sels_tmp = (float*) malloc(n*sizeof(float));
    memcpy(sels_tmp, sels, n*sizeof(float));

    for (int idx_l = 0; idx_l < s; idx_l++){
        RelationID id_l = expandToMask(RelationID::nth(idx_l), *eq_class); 
        RelationID new_id_l = remapRelid(id_l, remap_table); 
        int new_idx_l = (new_id_l.allLower() & new_eq_class).size();

        for (int idx_r = idx_l+1; idx_r < s; idx_r++){
            RelationID id_r = expandToMask(RelationID::nth(idx_r), *eq_class); 
            RelationID new_id_r = remapRelid(id_r, remap_table); 
            int new_idx_r = (new_id_r.allLower() & new_eq_class).size();

            int old_idx = eqClassIndex(idx_l, idx_r, s);
            int new_idx = eqClassIndex(new_idx_l, new_idx_r, s);

            sels[new_idx] = sels_tmp[old_idx];
        }
    }
    free(sels_tmp);
}

void remapPlannerInfo(GpuqoPlannerInfo* info, int* remap_table){
    remapBaseRels(info->base_rels, info->n_rels, remap_table);
    remapEdgeTable(info->edge_table, info->n_rels, remap_table);
    remapEdgeTable(info->indexed_edge_table, info->n_rels, remap_table);

    // even though it's not an edge_table, it has the same format
    if (gpuqo_spanning_tree_enable)
        remapEdgeTable(info->subtrees, info->n_rels, remap_table);

    size_t offset = 0;
    for (int i=0; i<info->n_eq_classes; i++){
        remapEqClass(&info->eq_classes[i], info->eq_class_sels+offset, remap_table);
		offset += eqClassNSels(info->eq_classes[i].size());
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
