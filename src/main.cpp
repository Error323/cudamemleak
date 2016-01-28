#include <iostream>
#include <vector>

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#define cudaSafeCall(expr) gpu::___cudaSafeCall(expr, __FILE__, __LINE__, __func__)

namespace gpu
{
void error(const char *error_string, const char *file, const int line, const char *func = "");

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        gpu::error(cudaGetErrorString(err), file, line, func);
}

void error(const char *error_string, const char *file, const int line, const char *func)
{
  std::cerr << "\n\nCudaError: " << error_string << std::endl;
  std::cerr << "  File:  " << file << std::endl;
  std::cerr << "  Line:  " << line << std::endl;
  std::cerr << "  Func:  " << func << std::endl;
  exit(1);
}

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
}

using namespace std;

int main(int argc, char *argv[])
{
  std::cout << "pid: " << getpid() << std::endl << std::endl;
  vector<int> data(512,0);
  gpu::memOp(data.data(), data.size());
  return 0;
}
