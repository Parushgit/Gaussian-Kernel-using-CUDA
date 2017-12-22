/*  YOUR_FIRST_NAME
 *  YOUR_LAST_NAME
 *  YOUR_UBIT_NAME
 */

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#ifndef A3_HPP
#define A3_HPP


using namespace std;


__global__ void Kernel(float *x_device , float *y_device , int num_size, float h) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Fetching the ID of each thread within a block
    const double pi = 3.14159;

    // __shared__ float *shared_x;
    // shared_x[idx] = x_device[idx];
    // __syncthreads();

    float sum = 0;
    if( idx < num_size)
    {
        for(int j = 0; j < num_size; j++)
        {
            sum = sum + (( 1 / sqrt(2 * pi)) * exp ( - ( ( pow ( ( ( x_device[idx] - x_device[j] ) / h ), 2) ) / 2)));
        }
        y_device[idx] = sum / (num_size * h);

    }
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) 
{
    int threads = 1024;

    int blocks = (n + threads - 1) / threads;

    float* x_device;
    float* y_device;

    cudaMalloc(&x_device,sizeof(float)* n); //Allocating memory for CUDA array x
    cudaMalloc(&y_device,sizeof(float)* n); //Allocating memory for CUDA array y

    cudaMemcpy(x_device, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice); 

    dim3 num_threads(threads);
    dim3 num_blocks(blocks);

    Kernel<<<num_blocks , num_threads>>>(x_device , y_device , n, h); 

    cudaMemcpy(y.data(), y_device, sizeof(float) * n, cudaMemcpyDeviceToHost);

} // gaussian_kde

//nvcc -std=c++11 -x cu a3.cpp -o a3


#endif // A3_HPP
