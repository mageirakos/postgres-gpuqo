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

// I did not want to include the full c.h for fear of conflicts so I just 
// include the definitions (to get USE_ASSERT_CHECKING) and just define the
// Assert macro as in c.h
#include "pg_config.h"
#ifndef USE_ASSERT_CHECKING
#define Assert(condition)	((void)true)
#else
#include <assert.h>
#define Assert(p) assert(p)
#endif

// I do the same for the CHECK_FOR_INTERRUPTS macro
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

// constants for table sizes
#define KB 1024ULL
#define MB (KB*1024)
#define GB (MB*1024)

// ceiled integer division
#define ceil_div(a,b) (((a)+(b)-1)/(b)) 

// structure representing join of two relations used by CUDA and CPU code 
// of GPUQO
struct JoinRelation{
	RelationID left_rel_id;
	RelationID right_rel_id;
	float rows;
	float cost;

public:
	__host__ __device__
	bool operator<(const JoinRelation &o) const
	{
		return cost < o.cost;
	}

	__host__ __device__
	bool operator>(const JoinRelation &o) const
	{
		return cost > o.cost;
	}

	__host__ __device__
	bool operator==(const JoinRelation &o) const
	{
		return cost == o.cost;
	}

	__host__ __device__
	bool operator<=(const JoinRelation &o) const
	{
		return cost <= o.cost;
	}

	__host__ __device__
	bool operator>=(const JoinRelation &o) const
	{
		return cost >= o.cost;
	}
};

struct JoinRelationDetailed : public JoinRelation{
	RelationID id;
	EdgeMask edges;
};

struct JoinRelationDpsize : public JoinRelationDetailed {
	uint32_t left_rel_idx;
	uint32_t right_rel_idx;
};

extern std::ostream & operator<<(std::ostream &os, const JoinRelation& jr);

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
