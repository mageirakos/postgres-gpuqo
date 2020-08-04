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

#include "nodes/pathnodes.h"

/* routines in gpuqo_main.c */
extern RelOptInfo *gpuqo(PlannerInfo *root,
						int number_of_rels, List *initial_rels);

extern void perform_stencil(float * a, float * b, const int N);

#endif							/* GPUQO_H */
