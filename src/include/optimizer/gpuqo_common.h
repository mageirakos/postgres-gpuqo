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
typedef unsigned int FixedBitMask;
typedef FixedBitMask EdgeMask;
typedef FixedBitMask RelationID;

typedef struct BaseRelation{
	RelationID id;
	unsigned int rows;
	unsigned int tuples;
	EdgeMask edges;
} BaseRelation;

typedef struct QueryTree{
	RelationID id;
	unsigned int rows;
	double cost;
	struct QueryTree* left;
	struct QueryTree* right;
} QueryTree;

#endif							/* GPUQO_COMMON_H */