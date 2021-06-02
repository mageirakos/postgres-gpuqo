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
#define TIME_DIFF_MS(end, begin) std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000000.0
#ifdef GPUQO_PROFILE
#define EXTERN_PROTOTYPE_TIMING(s)  		extern time_point timeStart_##s; extern double timeDiff_##s; extern double timeTotal_##s; extern int count_##s; extern double timeCheckpoint_##s; extern int countCheckpoint_##s; extern bool is_nv_##s
#define PROTOTYPE_TIMING_CLASS(s)  		time_point timeStart_##s; double  timeDiff_##s; double timeTotal_##s; int count_##s; double timeCheckpoint_##s; int countCheckpoint_##s; bool is_nv_##s
#define PROTOTYPE_TIMING(s)  		time_point  __attribute__((unused)) timeStart_##s; double  __attribute__((unused)) timeDiff_##s; double  __attribute__((unused)) timeTotal_##s; int  __attribute__((unused)) count_##s; double  __attribute__((unused)) timeCheckpoint_##s; int  __attribute__((unused)) countCheckpoint_##s; bool  __attribute__((unused)) is_nv_##s
#define INIT_TIMING(s)  		timeTotal_##s = 0; count_##s = 0; timeCheckpoint_##s = 0; countCheckpoint_##s = 0; is_nv_##s=false
#define INIT_NV_TIMING(s)  		timeTotal_##s = 0; count_##s = 0; timeCheckpoint_##s = 0; countCheckpoint_##s = 0; is_nv_##s=true
#define DECLARE_TIMING(s)  		PROTOTYPE_TIMING(s); INIT_TIMING(s)
#define DECLARE_NV_TIMING(s)  		PROTOTYPE_TIMING(s); INIT_NV_TIMING(s)
#define START_TIMING(s)    		timeStart_##s = NOW()
#define STOP_TIMING(s)     		if (is_nv_##s) cudaDeviceSynchronize(); timeDiff_##s = TIME_DIFF_MS(NOW(), timeStart_##s); timeTotal_##s += timeDiff_##s; count_##s++
#define CLEAR_TIMING(s)         timeTotal_##s = 0; count_##s = 0; timeCheckpoint_##s=timeTotal_##s; countCheckpoint_##s = count_##s
#define PRINT_TIMING(s) 		std::cout << #s " took " << (timeDiff_##s) << " ms" << std::endl
#define PRINT_TOTAL_TIMING(s)   std::cout << #s " took " << (double)(count_##s ? timeTotal_##s : 0) << " ms in total over " << count_##s << " runs" << std::endl; timeCheckpoint_##s=timeTotal_##s; countCheckpoint_##s = count_##s
#define PRINT_CHECKPOINT_TIMING(s)   std::cout << #s " took " << (double)(count_##s - countCheckpoint_##s ? timeTotal_##s - timeCheckpoint_##s : 0) << "ms in total over " << count_##s - countCheckpoint_##s << " runs" << std::endl; timeCheckpoint_##s=timeTotal_##s; countCheckpoint_##s = count_##s
#define PRINT_AVERAGE_TIMING(s) std::cout << #s " took " << (timeTotal_##s) / ((double)count_##s) << " ms in average over " << count_##s << " runs" << std::endl; timeCheckpoint_##s=timeTotal_##s; countCheckpoint_##s = count_##s
#else  /* ifdef GPUQO_PROFILE */
#define EXTERN_PROTOTYPE_TIMING(s)
#define PROTOTYPE_TIMING_CLASS(s)
#define PROTOTYPE_TIMING(s)
#define INIT_TIMING(s)
#define INIT_NV_TIMING(s)
#define DECLARE_TIMING(s)
#define DECLARE_NV_TIMING(s)
#define START_TIMING(s)
#define STOP_TIMING(s)
#define CLEAR_TIMING(s)
#define PRINT_TIMING(s)
#define PRINT_TOTAL_TIMING(s)
#define PRINT_CHECKPOINT_TIMING(s)
#define PRINT_AVERAGE_TIMING(s)
#endif /* ifdef GPUQO_PROFILE */
	
#endif							/* GPUQO_TIMING_CUH */
