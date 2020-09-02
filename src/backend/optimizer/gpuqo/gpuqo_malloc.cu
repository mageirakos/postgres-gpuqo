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
