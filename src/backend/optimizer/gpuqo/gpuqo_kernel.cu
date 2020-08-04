#include <stdio.h>

__global__ void kernel(float * a, float * b, const int N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (1 < i && 1 < j && 1 < k && i < N - 1 && j < N - 1 && k < N - 1) {
        a[i * N + j * N + k] = 0.8 * (b[(i - 1) * N + j * N + k] + b[(i + 1) * N + N * j + k] + b[i * N + (j - 1) * N + k] +
                               b[i * N + (j + 1) * N + k] + b[i * N + j * N + k-1] + b[i * N + j * N  + k + 1]);
    }
}


extern "C" void perform_stencil(float * a, float * b, const int N) {
    float * d_a;
    float * d_b;

    cudaEvent_t start, stop;
    float       elapsedTime;

    /* begin timing */
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    cudaMalloc(&d_a, sizeof(float) * N * N * N);
    cudaMalloc(&d_b, sizeof(float) * N * N * N);

    cudaMemcpy(d_a, a, sizeof(float) * N * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N * N * N, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y, N / threadsPerBlock.z);

    kernel <<<numBlocks, threadsPerBlock>>>(d_a, d_b, N);
    cudaMemcpy(a,d_a, sizeof(float) * N * N * N, cudaMemcpyDeviceToHost);
    /* end timing */
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution time: %f seconds\n", elapsedTime / 1000);
    cudaFree(d_a);
    cudaFree(d_b);
}