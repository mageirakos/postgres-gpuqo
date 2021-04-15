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
template<typename BitmapsetN>
struct JoinRelation{
	BitmapsetN left_rel_id;
	BitmapsetN right_rel_id;
	float rows;
	float cost;

public:
	__host__ __device__
	bool operator<(const JoinRelation<BitmapsetN> &o) const
	{
		return cost < o.cost;
	}

	__host__ __device__
	bool operator>(const JoinRelation<BitmapsetN> &o) const
	{
		return cost > o.cost;
	}

	__host__ __device__
	bool operator==(const JoinRelation<BitmapsetN> &o) const
	{
		return cost == o.cost;
	}

	__host__ __device__
	bool operator<=(const JoinRelation<BitmapsetN> &o) const
	{
		return cost <= o.cost;
	}

	__host__ __device__
	bool operator>=(const JoinRelation<BitmapsetN> &o) const
	{
		return cost >= o.cost;
	}
};

template<typename BitmapsetN>
struct JoinRelationDetailed : public JoinRelation<BitmapsetN>{
	BitmapsetN id;
	BitmapsetN edges;
};

template<typename BitmapsetN>
struct JoinRelationDpsize : public JoinRelationDetailed<BitmapsetN> {
	uint_t<BitmapsetN> left_rel_idx;
	uint_t<BitmapsetN> right_rel_idx;
};

template<typename BitmapsetN>
struct QueryTree{
	BitmapsetN id;
	float rows;
	float cost;
	struct QueryTree* left;
	struct QueryTree* right;
};

template<typename BitmapsetN>
struct BaseRelation{
	BitmapsetN id;
	float rows;
	float tuples;
	int off_fk_selecs;
	int n_fk_selecs;
};

template<typename BitmapsetN>
struct GpuqoPlannerInfo{
	unsigned int size;
	int n_rels;
	BaseRelation<BitmapsetN> base_rels[BitmapsetN::SIZE];
	BitmapsetN edge_table[BitmapsetN::SIZE];
	BitmapsetN indexed_edge_table[BitmapsetN::SIZE];
	BitmapsetN subtrees[BitmapsetN::SIZE];
	int n_fk_selecs;
	unsigned int* fk_selec_idxs;
	float* fk_selec_sels;
	int n_eq_classes;
	int n_eq_class_sels;
	BitmapsetN* eq_classes;
	float* eq_class_sels;
};

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* 
convertGpuqoPlannerInfo(gpuqo_c::GpuqoPlannerInfo *info_c);

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* 
copyToDeviceGpuqoPlannerInfo(GpuqoPlannerInfo<BitmapsetN> *info);

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* 
deleteGpuqoPlannerInfo(GpuqoPlannerInfo<BitmapsetN> *info);

template<typename BitmapsetN>
gpuqo_c::QueryTree* convertQueryTree(QueryTree<BitmapsetN> *info);

#endif							/* GPUQO_PLANNER_INFO_CUH */
