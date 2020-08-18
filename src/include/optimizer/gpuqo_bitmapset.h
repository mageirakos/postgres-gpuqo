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

// For the moment it's limited to 64 relations
// I need to find a way to efficiently and dynamically increase this value
typedef unsigned long long Bitmapset64;

#define BMS64_EMPTY (0ULL)
#define BMS64_NTH(n) (1ULL<<(n))
#define BMS64_CMP(a) (~(a))
#define BMS64_LOWEST(a) ((a) & (-(a)))
#define BMS64_SET_ALL_LOWER(a) (a != BMS64_EMPTY ? BMS64_LOWEST(a)-1 : BMS64_EMPTY)
#define BMS64_SET_ALL_LOWER_INC(a) BMS64_UNION((a), BMS64_SET_ALL_LOWER(a))
#define BMS64_UNION(a, b) ((a) | (b))
#define BMS64_INTERSECTION(a, b) ((a) & (b))
#define BMS64_INTERSECTS(a, b) (BMS64_INTERSECTION((a), (b)) != BMS64_EMPTY)
#define BMS64_DIFFERENCE(a, b) BMS64_INTERSECTION((a), BMS64_CMP(b))
#define BMS64_NEXT_SUBSET(subset, set) ((set) & ((subset)-(set)))
#define BMS64_SET(a, n) BMS64_UNION((a), BMS64_NTH(n))
#define BMS64_UNSET(a, n) BMS64_DIFFERENCE((a), BMS64_NTH(n))
#define BMS64_IS_SET(a, n) BMS64_INTERSECTS((a), BMS64_NTH(n))
#define BMS64_EXPAND_TO_MASK(val, mask) _pdep_u64(val, mask)

#ifdef __CUDA_ARCH__
#define BMS64_SIZE(s) __popcll(s)
#define BMS64_LOWEST_POS(s) __ffsll(s)
#define BMS64_HIGHEST_POS(s) (64-__clzll(s))
#else
#define BMS64_SIZE(s) __builtin_popcount(s)
#define BMS64_LOWEST_POS(s) __builtin_ffsll(s)
#define BMS64_HIGHEST_POS(s) (64-__builtin_clzll(s))
#endif

#ifndef __BMI2__
#ifdef __CUDA_ARCH__
__device__ __host__
#endif
inline uint64_t _pdep_u64(uint64_t val, uint64_t mask){
    uint64_t res = 0ULL;
    for (uint64_t bb = 1; mask; bb += bb) {
        if (val & bb)
            res |= mask & -mask;
        mask &= mask - 1;
  }
  return res;
}
#endif
	
#endif							/* GPUQO_BITMAPSET_H */
