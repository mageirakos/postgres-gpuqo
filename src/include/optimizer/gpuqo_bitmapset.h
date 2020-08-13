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
#define BMS64_LOWEST(a) ((a) & (-(a)))
#define BMS64_SET_ALL_LOWER(a) (a != BMS64_EMPTY ? BMS64_LOWEST(a)-1 : BMS64_EMPTY)
#define BMS64_UNION(a, b) ((a) | (b))
#define BMS64_INTERSECTION(a, b) ((a) & (b))
#define BMS64_INTERSECTS(a, b) (((a) & (b)) != BMS64_EMPTY)
#define BMS64_DIFFERENCE(a, b) ((a) - BMS64_INTERSECTION((a), (b)))
#define BMS64_NEXT_SUBSET(subset, set) ((set) & ((subset)-(set)))

#ifdef __CUDA_ARCH__
#define BMS64_SIZE(s) __popcll(s)
#else
#define BMS64_SIZE(s) __builtin_popcount(s)
#endif
	
#endif							/* GPUQO_BITMAPSET_H */
