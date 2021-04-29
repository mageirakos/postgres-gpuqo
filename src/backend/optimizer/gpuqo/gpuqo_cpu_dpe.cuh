/*-------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpe.cuh
 *	  Definition of the functions implementing the parallel CPU algorithm DPE
 *
 * src/include/optimizer/gpuqo_cpu_dpe.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_CPU_DPE_CUH
#define GPUQO_CPU_DPE_CUH

#include "gpuqo_cpu_common.cuh"

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_cpu_dpe(GpuqoPlannerInfo<BitmapsetN>* info, CPUAlgorithm<BitmapsetN, hashtable_memo_t<BitmapsetN> > *algorithm);

#endif							/* GPUQO_CPU_DPE_CUH */
