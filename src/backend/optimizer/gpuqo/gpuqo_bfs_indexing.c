/*------------------------------------------------------------------------
 *
 * gpuqo_bfs_indexing.c
 *	  utilities to remap indices so that they are BFS-coherent
 *
 * src/backend/optimizer/gpuqo/gpuqo_main.c
 *
 *-------------------------------------------------------------------------
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "optimizer/gpuqo_common.h"

void makeBFSIndexRemapTables(int *remap_table_fw, int *remap_table_bw, GpuqoPlannerInfo* info){
    int bfs_idx = 0;
    Bitmapset32 seen = BMS32_EMPTY;
    int bfs_queue[32];
    int bfs_queue_idx = 0;
    
    bfs_queue[bfs_queue_idx++] = 0;

    while (bfs_queue_idx != 0 && bfs_idx < info->n_rels){
        int base_rel_idx = bfs_queue[--bfs_queue_idx];
        BaseRelation* r = &info->base_rels[base_rel_idx];
        EdgeMask edges = info->edge_table[base_rel_idx];
        if (!BMS32_INTERSECTS(seen, r->id)){
            remap_table_fw[base_rel_idx] = bfs_idx;
            remap_table_bw[bfs_idx] = base_rel_idx;
            printf("idx   %d -> %d\n", base_rel_idx, bfs_idx);
            bfs_idx++;
        
            while (edges != BMS32_EMPTY){
                int next = BMS32_LOWEST_POS(edges) - 2;
                bfs_queue[bfs_queue_idx++] = next;
                edges = BMS32_DIFFERENCE(edges, BMS32_LOWEST(edges));
            }
            seen = BMS32_UNION(seen, r->id);
        }
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
    printf("relid %u -> %u\n", in, out);
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

    EqClassInfo* p = info->eq_classes;
    while (p != NULL){
        p->relids = remapRelid(p->relids, remap_table);
        p = p->next;
    }
}

void remapQueryTree(QueryTree* qt, int* remap_table){
    if (qt == NULL)
        return;

    qt->id = remapRelid(qt->id, remap_table);

    remapQueryTree(qt->left, remap_table);
    remapQueryTree(qt->right, remap_table);
}
