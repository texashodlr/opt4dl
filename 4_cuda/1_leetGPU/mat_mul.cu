#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

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
    float* xD, *yD, *zD;

    x = (float*)malloc((M*N)*sizeof(float));
    y = (float*)malloc((N*K)*sizeof(float));
    z = (float*)malloc((M*K)*sizeof(float));

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

    // Capture start time
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate( &start));
    HANDLE_ERROR( cudaEventCreate( &stop));
    HANDLE_ERROR( cudaEventRecord( start, 0));

    HANDLE_ERROR(cudaMalloc((void**)&xD, (M*N) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&yD, (N*K) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&zD, (M*K) * sizeof(float)));
    
    HANDLE_ERROR( cudaMemcpy(xD, x, (M*N)*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(yD, y, (N*K)*sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(xD, yD, zD, M, N, K);
    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy(z, zD, (M*K)*sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR( cudaEventRecord(stop, 0));
    HANDLE_ERROR( cudaEventSynchronize(stop));
    float elapsed_time;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsed_time, start, stop));
    printf( "Time to generate:  %3.1f ms\n",elapsed_time);

    //float maxError = 0.0f;
    for (int i = 0; i < (M*K); i++){
        //maxError = fmax(maxError, fabs(z[i] - 3.0f));
        std::cout << "Element[" << i << "] = Z: " << z[i] << std::endl;
    }
    

    HANDLE_ERROR(cudaFree(xD));
    HANDLE_ERROR(cudaFree(yD));
    HANDLE_ERROR(cudaFree(zD));

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    free(x);
    free(y);
    free(z);
    

    return 0;

}