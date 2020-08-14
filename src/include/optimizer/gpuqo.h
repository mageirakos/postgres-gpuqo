/*-------------------------------------------------------------------------
 *
 * gpuqo.h
 *	  prototypes for gpuqo_main.c
 *
 * src/include/optimizer/gpuqo.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_H
#define GPUQO_H

#include "optimizer/gpuqo_common.h"

#include "nodes/pathnodes.h"

/* routines in gpuqo_main.c */
extern RelOptInfo *gpuqo(PlannerInfo *root,
						int n_rels, List *initial_rels);

extern QueryTree* gpuqo_dpsize(BaseRelation base_rels[], int n_rels, EdgeInfo* edge_table);
extern QueryTree* gpuqo_cpu_dpsize(BaseRelation base_rels[], int n_rels, EdgeInfo* edge_table);
extern QueryTree* gpuqo_cpu_dpsub(BaseRelation base_rels[], int n_rels, EdgeInfo* edge_table);
extern QueryTree* gpuqo_cpu_dpccp(BaseRelation base_rels[], int n_rels, EdgeInfo* edge_table);
extern QueryTree* gpuqo_dpe_dpsize(BaseRelation base_rels[], int n_rels, EdgeInfo* edge_table);
extern QueryTree* gpuqo_dpe_dpsub(BaseRelation base_rels[], int n_rels, EdgeInfo* edge_table);
extern QueryTree* gpuqo_dpe_dpccp(BaseRelation base_rels[], int n_rels, EdgeInfo* edge_table);

extern bool gpuqo_check_can_run(PlannerInfo* root);

#endif							/* GPUQO_H */
