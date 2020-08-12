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

typedef enum GpuqoAlgorithm {
	GPUQO_DPSIZE,
	GPUQO_CPU_DPSIZE
} GpuqoAlgorithm;

extern GpuqoAlgorithm gpuqo_algorithm;
extern int gpuqo_dpsize_min_scratchpad_size_mb;
extern int gpuqo_dpsize_max_scratchpad_size_mb;
extern int gpuqo_dpsize_max_memo_size_mb;

// For the moment it's limited to 64 relations
// I need to find a way to efficiently and dynamically increase this value
typedef unsigned long long FixedBitMask;
typedef FixedBitMask EdgeMask;
typedef FixedBitMask RelationID;

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
