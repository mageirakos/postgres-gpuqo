/*-------------------------------------------------------------------------
 *
 * gpuqo_timing.cuh
 *	  timing macros for CUDA
 *
 * src/include/optimizer/gpuqo_timing.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_TIMING_CUH
#define GPUQO_TIMING_CUH

#include <chrono>

// activate profiling as a conequence of debug (if not yet active)
#ifdef GPUQO_DEBUG
#ifndef GPUQO_PROFILE
#define GPUQO_PROFILE
#endif
#endif

typedef std::chrono::steady_clock::time_point time_point;
#define NOW() std::chrono::steady_clock::now()
#define TIME_DIFF_MS(end, begin) std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0
#ifdef GPUQO_PROFILE
#define DECLARE_TIMING(s)  		time_point timeStart_##s; double timeDiff_##s; double timeTotal_##s = 0; int count_##s = 0
#define START_TIMING(s)    		timeStart_##s = NOW()
#define STOP_TIMING(s)     		cudaThreadSynchronize(); timeDiff_##s = TIME_DIFF_MS(NOW(), timeStart_##s); timeTotal_##s += timeDiff_##s; count_##s++
#define CLEAR_AVERAGE_TIMING(s) timeTotal_##s = 0; count_##s = 0
#define PRINT_TIMING(s) 		std::cout << #s " took " << (timeDiff_##s) << "ms" << std::endl
#define PRINT_TOTAL_TIMING(s)   std::cout << #s " took " << (double)(count_##s ? timeTotal_##s : 0) << "ms in total over " << count_##s << " runs" << std::endl
#define PRINT_AVERAGE_TIMING(s) std::cout << #s " took " << (timeTotal_##s) / ((double)count_##s) << "ms in average over " << count_##s << " runs" << std::endl
#else  /* ifdef GPUQO_PROFILE */
#define DECLARE_TIMING(s)
#define START_TIMING(s)
#define STOP_TIMING(s)
#define CLEAR_AVERAGE_TIMING(s)
#define PRINT_TIMING(s)
#define PRINT_TOTAL_TIMING(s)
#define PRINT_AVERAGE_TIMING(s)
#endif /* ifdef GPUQO_PROFILE */
	
#endif							/* GPUQO_TIMING_CUH */
