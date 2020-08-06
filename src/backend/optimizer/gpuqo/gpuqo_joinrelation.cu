/*------------------------------------------------------------------------
 *
 * gpuqo_joinrelation.c
 *      definition of JoinRelation-related functions
 *
 * src/backend/optimizer/gpuqo/gpuqo_joinrelation.c
 *
 *-------------------------------------------------------------------------
 */

#include <cmath>
#include <iostream>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"

__host__
std::ostream & operator<<(std::ostream &os, const JoinRelation& jr)
{
	os<<"("<<jr.left_relation_idx<<","<<jr.right_relation_idx;
	os<<"): rows="<<jr.rows<<", cost="<<jr.cost;
	return os;
}
