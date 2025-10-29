#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float Pvalue = 0;

    if(y >= M || x >= K){return;}

    for(int k = 0; k < N; ++k){
        Pvalue += A[y*N + k] * B[k*K + x];
    }
    C[y*K + x] = Pvalue;
}


int main(void){
    int M = 3; //8192;
    int N = 2; //6144;
    int K = 3; //4096;
    // A = MxN
    // B = NxK
    // C = MxK
    float* x, *y, *z;
    cudaMallocManaged(&x, (M*N) * sizeof(float));
    cudaMallocManaged(&y, (N*K) * sizeof(float));
    cudaMallocManaged(&z, (M*K) * sizeof(float));

    for(int i=0; i < M; i++){
        for(int j=0; j < N; j++){
            x[i*N + j] = j + i*M;
        }
    }

    for(int i=0; i < N; i++){
        for(int j=0; j < K; j++){
            y[i*K + j] = j + i*N;
        }
    }

    for(int i=0; i < M; i++){
        for(int j=0; j < K; j++){
            z[i*K + j] = 0;
        }
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, z, M, N, K);

    cudaDeviceSynchronize();

    //float maxError = 0.0f;
    for (int i = 0; i < (M*K); i++){
        //maxError = fmax(maxError, fabs(z[i] - 3.0f));
        std::cout << "Element[" << i << "] = Z: " << z[i] << std::endl;
    }
    

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;

}