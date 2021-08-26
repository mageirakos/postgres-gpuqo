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

template<typename BitmapsetN>
Remapper<BitmapsetN,BitmapsetN> makeBFSIndexRemapper(GpuqoPlannerInfo<BitmapsetN>* info)
{
    int *bfs_queue = new int[info->n_rels];
    int bfs_queue_front_idx = 0;
    int bfs_queue_back_idx = 0;
    list<remapper_transf_el_t<BitmapsetN>> remap_list;

    int bfs_idx = 0;
    
    bfs_queue[bfs_queue_back_idx++] = 0;

    BitmapsetN seen = BitmapsetN::nth(1);
    // Look at this to understand how to iterate through the graph (BFS through graph)
    while (bfs_queue_front_idx != bfs_queue_back_idx && bfs_idx < info->n_rels){
        int base_rel_idx = bfs_queue[bfs_queue_front_idx++];
        
        BitmapsetN edges = info->edge_table[base_rel_idx];

        remapper_transf_el_t<BitmapsetN> remap_el;
        remap_el.from_relid=BitmapsetN::nth(base_rel_idx+1);
        remap_el.to_idx=bfs_idx;
        remap_el.qt=NULL;
        remap_list.push_back(remap_el);
        bfs_idx++;

        while (!edges.empty()){
            int next = edges.lowestPos();
            Assert(next > 0);

            if (!seen.isSet(next)){
                bfs_queue[bfs_queue_back_idx++] = next - 1;
            }
            
            edges.unset(next);
        }
        seen |= info->edge_table[base_rel_idx];
    }

    delete[] bfs_queue;
    return Remapper<BitmapsetN,BitmapsetN>(remap_list);
}

template Remapper<Bitmapset32,Bitmapset32> makeBFSIndexRemapper<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template Remapper<Bitmapset64,Bitmapset64> makeBFSIndexRemapper<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
template Remapper<BitmapsetDynamic,BitmapsetDynamic> makeBFSIndexRemapper<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic>*);
