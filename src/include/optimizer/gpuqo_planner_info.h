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

#include <nodes/bitmapset.h>

typedef struct EquivalenceClass EquivalenceClass;

typedef Bitmapset* EdgeMask;
typedef Bitmapset* RelationID;


typedef struct VarStat {
	float stadistinct;
	float stanullfrac;
	float mcvfreq;
} VarStat;

typedef struct EqClassInfo{
	EquivalenceClass* eclass;
	RelationID relids;
	float* sels;
	VarStat* stats;
	RelationID* fk;
	struct EqClassInfo* next;
} EqClassInfo;

typedef struct BaseRelation{
	RelationID id;
	float rows;
	float tuples;
	float pages;
	struct {
		float startup;
		float total;
	} cost;
	int width;
} BaseRelation;

typedef struct GpuqoPlannerInfo{
	int n_rels;
	BaseRelation *base_rels;
	EdgeMask* edge_table;
	EdgeMask* indexed_edge_table;
	EqClassInfo* eq_classes;
    int n_eq_classes;
    int n_eq_class_sels;
    int n_eq_class_fks;
    int n_eq_class_stats;
} GpuqoPlannerInfo;

typedef struct QueryTree{
	RelationID id;
	float rows;
	struct {
		float startup;
		float total;
	} cost;
	int width;
	struct QueryTree* left;
	struct QueryTree* right;
} QueryTree;
	
#endif							/* GPUQO_PLANNER_INFO_H */
