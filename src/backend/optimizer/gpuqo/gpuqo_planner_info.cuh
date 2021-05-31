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
	float cost;
	bool composite;
};

template<typename BitmapsetN>
struct GpuqoPlannerInfo{
	unsigned int size;
	int n_rels;
	int n_iters;
	BaseRelation<BitmapsetN> base_rels[BitmapsetN::SIZE];
	BitmapsetN edge_table[BitmapsetN::SIZE];
	BitmapsetN indexed_edge_table[BitmapsetN::SIZE];
	BitmapsetN subtrees[BitmapsetN::SIZE];
	struct {
		int n;
		BitmapsetN* relids;
		int n_sels;
		float* sels;
		int n_fks;
		BitmapsetN* fks; 
	} eq_classes;
};

__host__ __device__
inline size_t align64(size_t size) {
	if (size & 7) {
		return (size & (~7)) + 8;
	} else {
		return size & (~7);
	}
}

template<typename BitmapsetN>
__host__ __device__
inline size_t plannerInfoBaseSize() {
	return align64(sizeof(GpuqoPlannerInfo<BitmapsetN>));
}

template<typename BitmapsetN>
__host__ __device__
inline size_t plannerInfoEqClassesSize(int n_eq_classes) {
	return align64(sizeof(BitmapsetN) * n_eq_classes);
}

template<typename BitmapsetN>
__host__ __device__
inline size_t plannerInfoEqClassSelsSize(int n_eq_class_sels) {
	return align64(sizeof(float) * n_eq_class_sels);
}

template<typename BitmapsetN>
__host__ __device__
inline size_t plannerInfoEqClassFksSize(int n_eq_class_fks) {
	return align64(sizeof(BitmapsetN) * n_eq_class_fks);
}

template<typename BitmapsetN>
__host__ __device__
inline size_t plannerInfoSize(size_t n_eq_classes, size_t n_eq_class_sels, 
						size_t n_eq_class_fks) 
{
	return plannerInfoBaseSize<BitmapsetN>() 
		+ plannerInfoEqClassesSize<BitmapsetN>(n_eq_classes)
		+ plannerInfoEqClassSelsSize<BitmapsetN>(n_eq_class_sels)
		+ plannerInfoEqClassFksSize<BitmapsetN>(n_eq_class_fks);
}

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
