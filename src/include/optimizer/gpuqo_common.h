/*-------------------------------------------------------------------------
 *
 * gpuqo_common.h
 *	  definitions for both C and CUDA code
 *
 * src/include/optimizer/gpuqo_common.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_COMMON_H
#define GPUQO_COMMON_H

#include <optimizer/gpuqo_bitmapset.h>

typedef enum GpuqoAlgorithm {
	GPUQO_DPSIZE = 0,
	GPUQO_DPSUB,
	GPUQO_CPU_DPSIZE,
	GPUQO_CPU_DPSUB,
	GPUQO_CPU_DPCCP,
	GPUQO_DPE_DPSIZE,
	GPUQO_DPE_DPSUB,
	GPUQO_DPE_DPCCP
} GpuqoAlgorithm;

extern int gpuqo_algorithm;
extern int gpuqo_scratchpad_size_mb;
extern int gpuqo_max_memo_size_mb;
extern int gpuqo_n_parallel;
extern bool gpuqo_dpsub_filter_enable;
extern int gpuqo_dpsub_filter_threshold;
extern int gpuqo_dpsub_filter_cpu_enum_threshold;
extern int gpuqo_dpsub_filter_keys_overprovisioning;
extern bool gpuqo_dpsub_csg_enable;
extern int gpuqo_dpsub_csg_threshold;
extern int gpuqo_dpe_n_threads;
extern int gpuqo_dpe_pairs_per_depbuf;

typedef Bitmapset32 EdgeMask;
typedef Bitmapset32 RelationID;

typedef struct EqClassInfo{
	RelationID relids;
	float* sels;
	struct EqClassInfo* next;
} EqClassInfo;

typedef struct BaseRelation{
	RelationID id;
	float rows;
	float tuples;
} BaseRelation;

typedef struct QueryTree{
	RelationID id;
	float rows;
	float cost;
	struct QueryTree* left;
	struct QueryTree* right;
} QueryTree;

typedef struct GpuqoPlannerInfo{
	int n_rels;
	BaseRelation *base_rels;
	EdgeMask* edge_table;
	EdgeMask* indexed_edge_table;
	EqClassInfo* eq_classes;
} GpuqoPlannerInfo;

#endif							/* GPUQO_COMMON_H */
