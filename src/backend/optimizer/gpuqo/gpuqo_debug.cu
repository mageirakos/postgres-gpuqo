/*------------------------------------------------------------------------
 *
 * gpuqo_debug.cu
 *      definition of functions for debug printing
 *
 * src/backend/optimizer/gpuqo/gpuqo_debug.cu
 *
 *-------------------------------------------------------------------------
 */

#include <bitset>

#include "gpuqo_debug.cuh"

__host__
std::ostream & operator<<(std::ostream &os, const uint2& idxs)
{
	os<<"["<<idxs.x<<","<<idxs.y<<"]";
	return os;
}

template <typename Type>
__host__
std::ostream & operator<<(std::ostream &os, const Bitmapset<Type>& bms){
    os<<"["<<std::bitset<N_TO_SHOW>(bms.toUint())<<"]";
    return os;
}

__host__
std::ostream & operator<<(std::ostream &os, const JoinRelation& jr)
{
	RelationID jr_id = jr.left_rel_id | jr.right_rel_id;
	os<<jr_id<<" ("<<jr.left_rel_id<<","<<jr.right_rel_id<<"):";
	os<<"rows="<<jr.rows<<", cost="<<jr.cost;
	return os;
}

template
__host__
std::ostream & operator<<(std::ostream &os, const Bitmapset<uint32_t>& bms);

template
__host__
std::ostream & operator<<(std::ostream &os, const Bitmapset<uint64_t>& bms);
