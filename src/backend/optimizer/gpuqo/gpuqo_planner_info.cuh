/*-------------------------------------------------------------------------
 *
 * gpuqo_planner_info.cuh
 *	  structure for planner info in GPU memory
 *
 * src/include/optimizer/gpuqo.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_PLANNER_INFO_CUH
#define GPUQO_PLANNER_INFO_CUH

#include <optimizer/gpuqo_common.h>
#include "gpuqo_bitmapset.cuh"

#define SIZEOF_VOID_P 8
#define FLEXIBLE_ARRAY_MEMBER /**/

namespace gpuqo_c{
	typedef uint64_t uint64;
	typedef int64_t int64;
	typedef uint32_t uint32;
	typedef int32_t int32;
	#include <optimizer/gpuqo_planner_info.h>
};

typedef Bitmapset32 EdgeMask;
typedef Bitmapset32 RelationID;

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

typedef struct QueryTree{
	RelationID id;
	float rows;
	float cost;
	struct QueryTree* left;
	struct QueryTree* right;
} QueryTree;


struct BaseRelation{
	RelationID id;
	float rows;
	float tuples;
	int off_fk_selecs;
	int n_fk_selecs;
};

struct GpuqoPlannerInfo{
	unsigned int size;
	int n_rels;
	BaseRelation base_rels[32];
	EdgeMask edge_table[32];
	EdgeMask indexed_edge_table[32];
	RelationID subtrees[32];
	int n_fk_selecs;
	unsigned int* fk_selec_idxs;
	float* fk_selec_sels;
	int n_eq_classes;
	int n_eq_class_sels;
	RelationID* eq_classes;
	float* eq_class_sels;
};

GpuqoPlannerInfo* convertGpuqoPlannerInfo(gpuqo_c::GpuqoPlannerInfo *info_c);
GpuqoPlannerInfo* copyToDeviceGpuqoPlannerInfo(GpuqoPlannerInfo *info);
GpuqoPlannerInfo* deleteGpuqoPlannerInfo(GpuqoPlannerInfo *info);

gpuqo_c::QueryTree* convertQueryTree(QueryTree *info);

#endif							/* GPUQO_PLANNER_INFO_CUH */
