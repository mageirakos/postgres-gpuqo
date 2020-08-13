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
	GPUQO_CPU_DPSIZE,
	GPUQO_CPU_DPCCP
} GpuqoAlgorithm;

extern int gpuqo_algorithm;
extern int gpuqo_dpsize_min_scratchpad_size_mb;
extern int gpuqo_dpsize_max_scratchpad_size_mb;
extern int gpuqo_dpsize_max_memo_size_mb;

typedef Bitmapset64 EdgeMask;
typedef Bitmapset64 RelationID;

typedef struct BaseRelation{
	RelationID id;
	double rows;
	double tuples;
	EdgeMask edges;
} BaseRelation;

typedef struct QueryTree{
	RelationID id;
	double rows;
	double cost;
	struct QueryTree* left;
	struct QueryTree* right;
} QueryTree;

typedef struct EdgeInfo{
	double sel;
} EdgeInfo;

#endif							/* GPUQO_COMMON_H */
