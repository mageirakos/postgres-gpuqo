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

extern QueryTree* gpuqo_cpu_dpe(GpuqoPlannerInfo* info, DPCPUAlgorithm algorithm);

#endif							/* GPUQO_CPU_DPE_CUH */
