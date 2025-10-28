#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

int main(void){
    int N = 1 << 20;
    float* x, *y, *z;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));

    for(int i=0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0.0f;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(x, y, z, N);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(z[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;

}