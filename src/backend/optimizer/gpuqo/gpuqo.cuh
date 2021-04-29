/*-------------------------------------------------------------------------
 *
 * gpuqo.cuh
 *	  function prototypes and struct definitions for CUDA/Thrust code
 *
 * src/include/optimizer/gpuqo.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_CUH
#define GPUQO_CUH

#include <iostream>
#include "optimizer/gpuqo_common.h"
#include "gpuqo_uninitalloc.cuh"
#include "gpuqo_planner_info.cuh"
#include "gpuqo_debug.cuh"

#include "signal.h"
extern "C" void ProcessInterrupts(void);
extern "C" volatile sig_atomic_t InterruptPending;

#define CHECK_FOR_INTERRUPTS() \
do { \
	if (InterruptPending) \
		ProcessInterrupts(); \
} while(0)

// floating-point infinity constants
#include <limits>
#ifdef __CUDA_ARCH__
#define INFD __longlong_as_double(0x7ff0000000000000ULL)
#define INFF __int_as_float(0x7f800000)
#define NAND __longlong_as_double(0xfff8000000000000ULL)
#define NANF __int_as_float(0x7fffffff)
#else
#define INFD std::numeric_limits<double>::infinity()
#define INFF std::numeric_limits<float>::infinity()
#define NAND std::numeric_limits<double>::nan()
#define NANF std::numeric_limits<float>::nan()
#endif

template<typename T>
using uninit_device_vector = thrust::device_vector<T, uninitialized_allocator<T> >;

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_dpsize(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_dpsub(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_cpu_dpsize(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_cpu_dpsub(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_cpu_dpsub_bicc(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_cpu_dpsub_bicc_parallel(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_cpu_dpccp(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_dpe_dpsize(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_dpe_dpsub(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_cpu_dpsub_parallel(GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern QueryTree<BitmapsetN>* gpuqo_dpe_dpccp(GpuqoPlannerInfo<BitmapsetN>* info);


template<typename BitmapsetN>
extern void makeBFSIndexRemapTables(int *remap_table_fw, int *remap_table_bw, GpuqoPlannerInfo<BitmapsetN>* info);

template<typename BitmapsetN>
extern void remapPlannerInfo(GpuqoPlannerInfo<BitmapsetN>* info, int* remap_table);

template<typename BitmapsetN>
extern void remapQueryTree(QueryTree<BitmapsetN>* qt, int* remap_table);

template<typename BitmapsetN>
extern void minimumSpanningTree(GpuqoPlannerInfo<BitmapsetN> *info);

template<typename BitmapsetN>
extern void buildSubTrees(BitmapsetN* subtrees, GpuqoPlannerInfo<BitmapsetN> *info);

#endif							/* GPUQO_CUH */
