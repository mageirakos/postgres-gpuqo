/*------------------------------------------------------------------------
 *
 * gpuqo_joinrelation.c
 *      definition of JoinRelation-related functions
 *
 * src/backend/optimizer/gpuqo/gpuqo_joinrelation.c
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo_debug.cuh"

__host__
std::ostream & operator<<(std::ostream &os, const uint2& idxs)
{
	os<<"["<<idxs.x<<","<<idxs.y<<"]";
	return os;
}
