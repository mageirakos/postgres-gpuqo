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
#include "gpuqo_bitmapset_dynamic.cuh"
#include "gpuqo_postgres.cuh"

// structure representing join of two relations used by CUDA and CPU code 
// of GPUQO
template<typename BitmapsetN>
struct JoinRelation{
	BitmapsetN left_rel_id;
	BitmapsetN right_rel_id;
	float rows;
	PathCost cost;
	int width;

public:
	__host__ __device__
	bool operator<(const JoinRelation<BitmapsetN> &o) const
	{
		if (cost.total == o.cost.total)
			return cost.startup < o.cost.startup;
		else
			return cost.total < o.cost.total;
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
	PathCost cost;
	int width;
	struct QueryTree<BitmapsetN>* left;
	struct QueryTree<BitmapsetN>* right;
};

template<typename BitmapsetN>
struct BaseRelation{
	BitmapsetN id;
	float rows;
	float tuples;
	int width;
	PathCost cost;
	bool composite;
};

struct GpuqoPlannerInfoParams {
		float effective_cache_size;
		float seq_page_cost;
		float random_page_cost;
		float cpu_tuple_cost;
		float cpu_index_tuple_cost;
		float cpu_operator_cost;
		float disable_cost;
		bool enable_seqscan;
		bool enable_indexscan;
		bool enable_tidscan;
		bool enable_sort;
		bool enable_hashagg;
		bool enable_nestloop;
		bool enable_mergejoin;
		bool enable_hashjoin;
		int work_mem;
};

template<typename BitmapsetN>
struct EqClasses {
		int n;
		BitmapsetN* relids;
		int n_sels;
		float* sels;
		int n_fks;
		BitmapsetN* fks; 
		int n_vars;
		VarInfo* vars;
};


template<typename BitmapsetN>
struct GpuqoPlannerInfo{
	unsigned int size;

	int n_rels;
	int n_iters;

	GpuqoPlannerInfoParams params;
	
	BaseRelation<BitmapsetN> base_rels[BitmapsetN::SIZE];
	BitmapsetN edge_table[BitmapsetN::SIZE];
	BitmapsetN subtrees[BitmapsetN::SIZE];

	EqClasses<BitmapsetN> eq_classes;
};

template<>
struct GpuqoPlannerInfo<BitmapsetDynamic>{
	unsigned int size;

	int n_rels;
	int n_iters;

	GpuqoPlannerInfoParams params;
	
	BaseRelation<BitmapsetDynamic> *base_rels;
	BitmapsetDynamic *edge_table;
	BitmapsetDynamic *subtrees;

	EqClasses<BitmapsetDynamic> eq_classes;
};

template<typename BitmapsetN>
static
void initGpuqoPlannerInfo(GpuqoPlannerInfo<BitmapsetN>* info) { }

template<> 
void initGpuqoPlannerInfo<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic>* info) { 
	info->base_rels = new BaseRelation<BitmapsetDynamic>[info->n_rels];
	info->edge_table = new BitmapsetDynamic[info->n_rels];
	info->subtrees = new BitmapsetDynamic[info->n_rels];
}

template<typename BitmapsetN>
static
void freeGpuqoPlannerInfo(GpuqoPlannerInfo<BitmapsetN>* info) {
	delete info;
}

template<>
void freeGpuqoPlannerInfo<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic>* info) {
	delete[] info->base_rels;
	delete[] info->edge_table;
	delete[] info->subtrees;
	delete info;
}

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
inline size_t plannerInfoEqClassVarsSize(int n_eq_class_vars) {
	return align64(sizeof(struct VarInfo) * n_eq_class_vars);
}

template<typename BitmapsetN>
__host__ __device__
inline size_t plannerInfoSize(size_t n_eq_classes, size_t n_eq_class_sels, 
						size_t n_eq_class_fks, size_t n_eq_class_vars) 
{
	return plannerInfoBaseSize<BitmapsetN>() 
		+ plannerInfoEqClassesSize<BitmapsetN>(n_eq_classes)
		+ plannerInfoEqClassSelsSize<BitmapsetN>(n_eq_class_sels)
		+ plannerInfoEqClassFksSize<BitmapsetN>(n_eq_class_fks)
		+ plannerInfoEqClassVarsSize<BitmapsetN>(n_eq_class_vars);
}

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* 
convertGpuqoPlannerInfo(GpuqoPlannerInfoC *info_c);

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* 
copyToDeviceGpuqoPlannerInfo(GpuqoPlannerInfo<BitmapsetN> *info);

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* 
deleteGpuqoPlannerInfo(GpuqoPlannerInfo<BitmapsetN> *info);

template<typename BitmapsetN>
QueryTreeC* convertQueryTree(QueryTree<BitmapsetN> *info);

#endif							/* GPUQO_PLANNER_INFO_CUH */
