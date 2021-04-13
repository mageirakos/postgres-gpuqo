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

#include "gpuqo_planner_info.cuh"

#include <iostream>

// I did not want to include the full c.h for fear of conflicts so I just 
// include the definitions (to get USE_ASSERT_CHECKING) and just define the
// Assert macro as in c.h
#include "pg_config.h"
#ifndef USE_ASSERT_CHECKING
#define Assert(condition)	((void)true)
#else
#include <assert.h>
#define Assert(p) assert(p)
#endif

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
    for(size_t mCount=offset; mIter != (to); ++mIter, ++mCount) \
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

__host__
std::ostream & operator<<(std::ostream &os, const uint2& idxs);

__host__
std::ostream & operator<<(std::ostream &os, const ulong2& idxs);

__host__
std::ostream & operator<<(std::ostream &os, const JoinRelation& jr);

template<typename Type>
__host__
std::ostream & operator<<(std::ostream &os, const Bitmapset<Type>& bms);

#endif							/* GPUQO_DEBUG_CUH */
