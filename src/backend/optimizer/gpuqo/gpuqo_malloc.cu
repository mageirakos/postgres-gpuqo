/*-------------------------------------------------------------------------
 *
 * gpuqo_malloc.cu
 *	  implementation of the gpuqo_malloc and gpuqo_free functions 
 *    using CUDA Unified Memory (cudaMallocManaged).
 *
 * src/include/optimizer/gpuqo_query_tree.cu
 *
 *-------------------------------------------------------------------------
 */

#include <optimizer/gpuqo.cuh>

bool using_gpu(){
    switch (gpuqo_algorithm)
    {
        // GPU execution
        case GPUQO_DPSIZE:
        case GPUQO_DPSUB:
            return true;
        
        // CPU execution
        default:
            return false;
    }
}

extern "C" void* gpuqo_malloc(size_t size){
    void *p = NULL;
    if (using_gpu()){
        cudaMallocManaged(&p, size);
    } else{
        p = malloc(size);
    }
    return p;
}

extern "C" void gpuqo_free(void* p){
    if (using_gpu()){
        cudaFree(p);
    } else{
        free(p);
    }
}

extern "C" void gpuqo_prefetch(GpuqoPlannerInfo* info){
    if (using_gpu()){
        cudaMemPrefetchAsync(info->base_rels, info->n_rels*sizeof(BaseRelation), 0, NULL);
        cudaMemPrefetchAsync(info->edge_table, info->n_rels*info->n_rels*sizeof(EdgeInfo), 0, NULL);
        cudaMemPrefetchAsync(info, sizeof(GpuqoPlannerInfo), 0, NULL);
    }
}