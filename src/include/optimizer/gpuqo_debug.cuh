/*-------------------------------------------------------------------------
 *
 * gpuqo_debug.cuh
 *	  debugginh macros and tools for CUDA
 *
 * src/include/optimizer/gpuqo_debug.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DEBUG_CUH
#define GPUQO_DEBUG_CUH

#define printVectorOffset(from, offset, to) { \
    auto mIter = (from); \
    mIter += (offset); \
    for(int mCount=offset; mIter != (to); ++mIter, ++mCount) \
        std::cout << mCount << " : " << *mIter << std::endl; \
}

#define printVector(from, to) printVectorOffset((from), 0, (to))
	
#endif							/* GPUQO_DEBUG_CUH */
