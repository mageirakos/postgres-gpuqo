/*-------------------------------------------------------------------------
 *
 * gpuqo_binomial.cuh
 *	  declaration of binomial-precoumputing functions and macros
 * 
 * src/include/optimizer/gpuqo_binomial.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_BINOMIAL_CUH
#define GPUQO_BINOMIAL_CUH

#include <thrust/device_vector.h>

void precompute_binoms(thrust::host_vector<uint32_t> &binoms, int N);

#define BINOM(binoms, N, n, k) (binoms)[(n)*((N)+1)+(k)]

#endif							/* GPUQO_BINOMIAL_CUH */
