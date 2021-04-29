/*-------------------------------------------------------------------------
 *
 * gpuqo_cpu_sequential.cuh
 *	  Definition of the functions implementing the generic sequential CPU alg
 *
 * src/include/optimizer/gpuqo_cpu_sequential.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_CPU_SEQUENTIAL_CUH
#define GPUQO_CPU_SEQUENTIAL_CUH

#include "gpuqo_cpu_common.cuh"

template<typename BitmapsetN, typename memo_t>
extern QueryTree<BitmapsetN>* gpuqo_cpu_sequential(GpuqoPlannerInfo<BitmapsetN>* info, CPUAlgorithm<BitmapsetN, memo_t> *algorithm);

#endif							/* GPUQO_CPU_SEQUENTIAL_CUH */
