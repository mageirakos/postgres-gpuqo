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

// activate profiling as a conequence of debug (if not yet active)
#ifdef GPUQO_DEBUG
#ifndef GPUQO_PROFILE
#define GPUQO_PROFILE
#endif
#endif

#ifdef GPUQO_DEBUG
#define LOG_DEBUG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define DUMP_VECTOR_OFFSET(from, offset, to) do { \
    auto mIter = (from); \
    mIter += (offset); \
    for(uint64_t mCount=offset; mIter != (to); ++mIter, ++mCount) \
        std::cout << mCount << " : " << *mIter << std::endl; \
} while(0)
#define DUMP_VECTOR(from, to) DUMP_VECTOR_OFFSET((from), 0, (to))
#else
#define LOG_DEBUG(fmt, ...) 
#define DUMP_VECTOR_OFFSET(from, offset, to)
#define DUMP_VECTOR(from, to)
#endif

#ifdef GPUQO_PROFILE
#define LOG_PROFILE(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define LOG_PROFILE(fmt, ...)
#endif

#endif							/* GPUQO_DEBUG_CUH */
