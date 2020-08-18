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
#include <bitset>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"

#define N_TO_SHOW 8

__host__
std::ostream & operator<<(std::ostream &os, const JoinRelation& jr)
{
	os<<"["<<std::bitset<N_TO_SHOW>(jr.id)<<"] ";
	os<<"("<<std::bitset<N_TO_SHOW>(jr.left_relation_id);
	os<<","<<std::bitset<N_TO_SHOW>(jr.right_relation_id)<<"):";
	os<<"rows="<<jr.rows<<", cost="<<jr.cost;
	return os;
}
