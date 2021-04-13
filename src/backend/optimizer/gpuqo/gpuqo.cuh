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

typedef thrust::device_vector<RelationID, uninitialized_allocator<RelationID> > uninit_device_vector_relid;
typedef thrust::device_vector<JoinRelation, uninitialized_allocator<JoinRelation> > uninit_device_vector_joinrel;
typedef thrust::device_vector<JoinRelationDpsize, uninitialized_allocator<JoinRelationDpsize> > uninit_device_vector_joinrel_dpsize;
typedef thrust::device_vector<uint2, uninitialized_allocator<uint2> > uninit_device_vector_uint2;

extern QueryTree* gpuqo_dpsize(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpsub(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_cpu_dpsize(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_cpu_dpsub(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_cpu_dpccp(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpe_dpsize(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpe_dpsub(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpe_dpccp(GpuqoPlannerInfo* info);

extern void makeBFSIndexRemapTables(int *remap_table_fw, int *remap_table_bw, GpuqoPlannerInfo* info);
extern RelationID remap_relid(RelationID id, int *remap_table);
extern void remapEdgeTable(EdgeMask* edge_table, int n, int* remap_table);
extern void remapPlannerInfo(GpuqoPlannerInfo* info, int* remap_table);
extern void remapQueryTree(QueryTree* qt, int* remap_table);

extern void minimumSpanningTree(GpuqoPlannerInfo *info);
extern void buildSubTrees(RelationID* subtrees, GpuqoPlannerInfo *info);

#endif							/* GPUQO_CUH */
