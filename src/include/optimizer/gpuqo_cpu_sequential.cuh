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

#include "optimizer/gpuqo_cpu_common.cuh"

extern QueryTree* gpuqo_cpu_sequential(BaseRelation base_rels[], int n_rels, EdgeInfo edge_table[], DPCPUAlgorithm algorithm);

#endif							/* GPUQO_CPU_SEQUENTIAL_CUH */
