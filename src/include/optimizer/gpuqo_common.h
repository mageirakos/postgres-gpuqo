/*-------------------------------------------------------------------------
 *
 * gpuqo.h
 *	  prototypes for gpuqo_main.c
 *
 * src/include/optimizer/gpuqo.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_COMMON_H
#define GPUQO_COMMON_H

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
