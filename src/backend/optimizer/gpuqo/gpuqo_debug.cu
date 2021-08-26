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
#include "gpuqo_planner_info.cuh"

__host__
std::ostream & operator<<(std::ostream &os, const uint2& idxs)
{
	os<<"["<<idxs.x<<","<<idxs.y<<"]";
	return os;
}

__host__
std::ostream & operator<<(std::ostream &os, const ulong2& idxs)
{
	os<<"["<<idxs.x<<","<<idxs.y<<"]";
	return os;
}

template <typename Type>
__host__
std::ostream & operator<<(std::ostream &os, const GpuqoBitmapset<Type>& bms){
    os<<"["<<std::bitset<N_TO_SHOW>(bms.toUlonglong())<<"]"; // use to be .toUint()
    return os;
}

__host__
std::ostream & operator<<(std::ostream &os, const BitmapsetDynamic& bms){
    os<<"["<<std::bitset<N_TO_SHOW>(bms.toUlonglong())<<"]";
    return os;
}

// uncomment when I want to use this operator
// std::ostream& operator<<(std::ostream &os, const BitmapsetDynamic& bms)  {
//         for (int i = bms.bms->nwords; i >= 0; i--)
//                 os<<std::bitset<64>(bms.bms->words[i]);
//         os<<std::endl;
//         return os;
// }

template
__host__
std::ostream & operator<<(std::ostream &os, const Bitmapset32& bms);

template
__host__
std::ostream & operator<<(std::ostream &os, const Bitmapset64& bms);

template<typename BitmapsetN>
__host__
std::ostream & operator<<(std::ostream &os, const JoinRelation<BitmapsetN>& jr)
{
	BitmapsetN jr_id = jr.left_rel_id | jr.right_rel_id;
	os<<jr_id<<" ("<<jr.left_rel_id<<","<<jr.right_rel_id<<"):";
	os<<"rows="<<jr.rows<<", cost="<<jr.cost.total;
	return os;
}

template
__host__
std::ostream & operator<<(std::ostream &os, const JoinRelation<Bitmapset32>& bms);

template
__host__
std::ostream & operator<<(std::ostream &os, const JoinRelation<Bitmapset64>& bms);
