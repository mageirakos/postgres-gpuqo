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
#include "optimizer/gpuqo_planner_info.h"

#include "postgres.h"
#include "nodes/pathnodes.h"

/* routines in gpuqo_main.c */
extern RelOptInfo *gpuqo(PlannerInfo *root,
						int n_rels, List *initial_rels);

extern QueryTree* gpuqo_run(int gpuqo_algorithm, GpuqoPlannerInfo* info);

extern bool gpuqo_check_can_run(PlannerInfo* root);

#endif							/* GPUQO_H */
