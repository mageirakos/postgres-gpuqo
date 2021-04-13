/*-------------------------------------------------------------------------
 *
 * gpuqo_binomial.cu
 *	  definition of binomial precomputing function
 *
 * src/include/optimizer/gpuqo_binomial.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo_binomial.cuh"

template<typename uint_t>
void precompute_binoms(thrust::host_vector<uint_t> &binoms, int N){
    for (int n = 0; n <= N; n++){
        for (int k = 0; k <= N; k++){
            int idx = n*(N+1)+k;
            if (k > n){
                // invalid
                binoms[idx] = 0;
            } else if (k == 0 || k == n){
                // base case
                binoms[idx] = 1;
            } else {
                binoms[idx] = binoms[(n-1)*(N+1)+(k-1)] + binoms[(n-1)*(N+1)+k];
            }
        }

    }
}

// explicit specification
template void precompute_binoms<unsigned int>(thrust::host_vector<unsigned int>&,int);
template void precompute_binoms<unsigned long long int>(thrust::host_vector<unsigned long long int>&,int);
