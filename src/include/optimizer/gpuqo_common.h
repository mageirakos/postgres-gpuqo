/*-------------------------------------------------------------------------
 *
 * gpuqo_common.h
 *	  definitions for both C and CUDA code
 *
 * src/include/optimizer/gpuqo_common.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_COMMON_H
#define GPUQO_COMMON_H

typedef enum GpuqoAlgorithm {
	GPUQO_DPSIZE = 0,
	GPUQO_DPSUB,
	GPUQO_CPU_DPSIZE,
	GPUQO_CPU_DPSUB,
	GPUQO_CPU_DPSUB_PARALLEL,
	GPUQO_CPU_DPSUB_BICC,
	GPUQO_CPU_DPSUB_BICC_PARALLEL,
	GPUQO_CPU_DPCCP,
	GPUQO_DPE_DPSIZE,
	GPUQO_DPE_DPSUB,
	GPUQO_DPE_DPCCP
} GpuqoAlgorithm;

extern int gpuqo_algorithm;
extern int gpuqo_scratchpad_size_mb;
extern int gpuqo_max_memo_size_mb;
extern int gpuqo_min_memo_size_mb;
extern int gpuqo_n_parallel;
extern bool gpuqo_dpsub_filter_enable;
extern int gpuqo_dpsub_filter_threshold;
extern int gpuqo_dpsub_filter_cpu_enum_threshold;
extern int gpuqo_dpsub_filter_keys_overprovisioning;
extern bool gpuqo_dpsub_ccc_enable;
extern bool gpuqo_dpsub_csg_enable;
extern int gpuqo_dpsub_csg_threshold;
extern int gpuqo_dpe_n_threads;
extern int gpuqo_cpu_dpsub_parallel_chunk_size;
extern int gpuqo_dpe_pairs_per_depbuf;
extern bool gpuqo_dpsub_tree_enable;
extern bool gpuqo_dpsub_bicc_enable;
extern bool gpuqo_spanning_tree_enable;

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
static inline int eqClassNSels(int size){
	return size*(size-1)/2;
}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
static inline int eqClassIndex(int idx_l, int idx_r, int size){
	if (idx_l < idx_r)
		return idx_l*size - idx_l*(idx_l+1)/2 + (idx_r-idx_l-1);
	else
		return eqClassIndex(idx_r, idx_l, size);
}

// constants for table sizes
#define KB 1024ULL
#define MB (KB*1024)
#define GB (MB*1024)

// ceiled integer division
#define ceil_div(a,b) (((a)+(b)-1)/(b)) 


#endif							/* GPUQO_COMMON_H */
