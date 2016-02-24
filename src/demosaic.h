#pragma once

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <stdint.h>

namespace gpu
{
class Demosaic
{
public:
  static Demosaic* Instance()
  {
    if (!sInstance)
      sInstance = new Demosaic();
    return sInstance;
  }

  static void Release()
  {
    if (sInstance)
      sInstance->~Demosaic();
  }

  void Initialize();
  void Process(const float *src, float *dst, int w, int h, float a);

private:
  Demosaic() {}
  ~Demosaic();
  static Demosaic *sInstance;

  cudaChannelFormatDesc chan_desc;
  cudaArray *cu_array;
};
} // namespace gpu
