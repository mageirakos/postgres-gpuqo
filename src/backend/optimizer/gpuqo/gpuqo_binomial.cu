/*-------------------------------------------------------------------------
 *
 * gpuqo_binomial.cu
 *	  definition of binomial precomputing function
 *
 * src/include/optimizer/gpuqo_binomial.cu
 *
 *-------------------------------------------------------------------------
 */

#include <optimizer/gpuqo_binomial.cuh>

void precompute_binoms(thrust::host_vector<uint64_t> &binoms, int N){
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
