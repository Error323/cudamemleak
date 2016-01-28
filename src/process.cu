#include <assert.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <stdint.h>
#include "init.cuh"

namespace gpu
{
  int memOp(int *data, int n)
  {
    int *d_data;

    size_t data_size = n * sizeof(int);
    cudaSafeCall(cudaMalloc((void**)&d_data, data_size));

    cudaSafeCall(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaDeviceSynchronize());

    for (int i = 0; i < 1000000; i++)
    {
      cudaSafeCall(cudaMemcpy(data, d_data, data_size, cudaMemcpyDeviceToHost));
      cudaSafeCall(cudaDeviceSynchronize());
    }

    cudaSafeCall(cudaFree(d_data));
    cudaSafeCall(cudaDeviceSynchronize());

    return 0;
  }
} // namespace gpu



