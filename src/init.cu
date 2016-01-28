#include "init.cuh"
#include <iostream>

#define DebugLine(x) std::cout << x << std::endl
#define ErrorLine(x) std::cerr << x << std::endl

#include <stdio.h>
#include <stdlib.h>

namespace gpu
{

bool initCuda(void)
{
  int device;
  bool has_cuda = (cudaGetDevice(&device)) == cudaSuccess;
  if (!has_cuda)
  {
    ErrorLine("Initializing cuda failed");
    return false;
  }

  cudaDeviceProp prop;
  cudaSafeCall(cudaGetDeviceProperties(&prop, device));

  DebugLine("");
  DebugLine("Initialized " << prop.name);
  DebugLine("  Compute capability:   " << prop.major << "." << prop.minor);
  DebugLine("  Multiprocessors:      " << prop.multiProcessorCount);
  DebugLine("  Global memory size:   " << prop.totalGlobalMem);
  DebugLine("  Constant memory size: " << prop.totalConstMem);
  DebugLine("  Concurrent kernels:   " << prop.concurrentKernels);
  DebugLine("  Shared mem per block: " << prop.sharedMemPerBlock);
  DebugLine("  Warpsize:             " << prop.warpSize);
  DebugLine("  Free/Total:           " << freeMemory() << "/" << totalMemory());
  DebugLine("");

  return true;
}

void error(const char *error_string, const char *file, const int line, const char *func)
{
  ErrorLine("\n\nCudaError: " << error_string);
  ErrorLine("  File:  " << file);
  ErrorLine("  Line:  " << line);
  ErrorLine("  Func:  " << func);
  exit(1);
}

size_t freeMemory()
{
  size_t free_memory, total_memory;
  cudaSafeCall(cudaMemGetInfo(&free_memory, &total_memory));
  return free_memory;
}

size_t totalMemory()
{
  size_t free_memory, total_memory;
  cudaSafeCall(cudaMemGetInfo(&free_memory, &total_memory));
  return total_memory;
}

bool smProcessorsAndCores(int &processors, int &cores)
{
  int device;
  bool has_cuda = (cudaGetDevice(&device)) == cudaSuccess;
  if (!has_cuda)
  {
    ErrorLine("Initializing cuda failed");
    return false;
  }

  cudaDeviceProp prop;
  cudaSafeCall(cudaGetDeviceProperties(&prop, device));
  processors = prop.multiProcessorCount;

  switch (prop.major)
  {
    case 1: cores = 8;                           return true;
    case 2: cores = (prop.minor == 0 ? 32 : 48); return true;
    case 3: cores = 192;                         return true;
    case 5: cores = 128;                         return true;
    default: ErrorLine("Unknown compute capability");
  }

  return false;
}

size_t sharedMemory()
{
  int device;
  bool has_cuda = (cudaGetDevice(&device)) == cudaSuccess;
  if (!has_cuda)
  {
    ErrorLine("Initializing cuda failed");
    return false;
  }

  cudaDeviceProp prop;
  cudaSafeCall(cudaGetDeviceProperties(&prop, device));

  return prop.sharedMemPerBlock;
}

int warpSize()
{
  int device;
  bool has_cuda = (cudaGetDevice(&device)) == cudaSuccess;
  if (!has_cuda)
  {
    ErrorLine("Initializing cuda failed");
    return false;
  }

  cudaDeviceProp prop;
  cudaSafeCall(cudaGetDeviceProperties(&prop, device));

  return prop.warpSize;
}
}
