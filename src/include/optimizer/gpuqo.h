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

extern QueryTree* gpuqo_dpsize(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpsub(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_cpu_dpsize(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_cpu_dpsub(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_cpu_dpccp(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpe_dpsize(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpe_dpsub(GpuqoPlannerInfo* info);
extern QueryTree* gpuqo_dpe_dpccp(GpuqoPlannerInfo* info);

extern bool gpuqo_check_can_run(PlannerInfo* root);
extern void* gpuqo_malloc(size_t size);
extern void gpuqo_free(void* p);

#endif							/* GPUQO_H */
