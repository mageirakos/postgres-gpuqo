/*-------------------------------------------------------------------------
 *
 * gpuqo_bitmapset.h
 *	  declaration of function manipulating bitmapsets.
 *
 * src/include/optimizer/gpuqo_bitmapset.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_BITMAPSET_H
#define GPUQO_BITMAPSET_H

#include "stdint.h"

// For the moment it's limited to 32 relations
// I need to find a way to efficiently and dynamically increase this value
typedef uint32_t Bitmapset32;

#define BMS32_EMPTY (0U)
#define BMS32_NTH(n) (1U<<(n))
#define BMS32_CMP(a) (~(a))
#define BMS32_LOWEST(a) ((a) & (-(a)))
#define BMS32_SET_ALL_LOWER(a) (a != BMS32_EMPTY ? BMS32_LOWEST(a)-1 : BMS32_EMPTY)
#define BMS32_SET_ALL_LOWER_INC(a) BMS32_UNION((a), BMS32_SET_ALL_LOWER(a))
#define BMS32_UNION(a, b) ((a) | (b))
#define BMS32_INTERSECTION(a, b) ((a) & (b))
#define BMS32_INTERSECTS(a, b) (BMS32_INTERSECTION((a), (b)) != BMS32_EMPTY)
#define BMS32_IS_SUBSET(a, b) (BMS32_INTERSECTION((a), (b)) == (a))
#define BMS32_DIFFERENCE(a, b) BMS32_INTERSECTION((a), BMS32_CMP(b))
#define BMS32_NEXT_SUBSET(subset, set) ((set) & ((subset)-(set)))
#define BMS32_SET(a, n) BMS32_UNION((a), BMS32_NTH(n))
#define BMS32_UNSET(a, n) BMS32_DIFFERENCE((a), BMS32_NTH(n))
#define BMS32_IS_SET(a, n) BMS32_INTERSECTS((a), BMS32_NTH(n))
#define BMS32_EXPAND_TO_MASK(val, mask) _pdep_u32(val, mask)

#ifdef __CUDA_ARCH__
#define BMS32_SIZE(s) __popc(s)
#define BMS32_LOWEST_POS(s) __ffs(s)
#define BMS32_HIGHEST_POS(s) (32-__clz(s))
#else
#define BMS32_SIZE(s) __builtin_popcount(s)
#define BMS32_LOWEST_POS(s) __builtin_ffs(s)
#define BMS32_HIGHEST_POS(s) (32-__builtin_clz(s))
#endif

#define BMS32_HIGHEST(a) (BMS32_NTH(BMS32_HIGHEST_POS(a)-1))

#ifndef __BMI2__
#ifdef __CUDA_ARCH__
__device__ __host__
#endif
inline uint32_t _pdep_u32(uint32_t val, uint32_t mask){
    uint32_t res = 0;
    for (uint32_t bb = 1; mask; bb += bb) {
        if (val & bb)
            res |= mask & -mask;
        mask &= mask - 1;
  }
  return res;
}
#endif
	
#endif							/* GPUQO_BITMAPSET_H */
