/*-------------------------------------------------------------------------
 *
 * gpuqo_planner_info.h
 *	  declaration of planner info for C files.
 *
 * src/include/optimizer/gpuqo_planner_info.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_PLANNER_INFO_H
#define GPUQO_PLANNER_INFO_H

#include <optimizer/gpuqo_bitmapset.h>

typedef Bitmapset32 EdgeMask;
typedef Bitmapset32 RelationID;

typedef struct EqClassInfo{
	RelationID relids;
	float* sels;
	struct EqClassInfo* next;
} EqClassInfo;

typedef struct FKSelecInfo{
	int other_baserel;
	float sel;
	struct FKSelecInfo* next;
} FKSelecInfo;

typedef struct BaseRelation{
	RelationID id;
	float rows;
	float tuples;
	FKSelecInfo* fk_selecs;
} BaseRelation;

typedef struct GpuqoPlannerInfo{
	int n_rels;
	BaseRelation *base_rels;
	EdgeMask* edge_table;
	EdgeMask* indexed_edge_table;
	EqClassInfo* eq_classes;
    int n_fk_selecs;
    int n_eq_classes;
    int n_eq_class_sels;
} GpuqoPlannerInfo;

typedef struct QueryTree{
	RelationID id;
	float rows;
	float cost;
	struct QueryTree* left;
	struct QueryTree* right;
} QueryTree;
	
#endif							/* GPUQO_PLANNER_INFO_H */
