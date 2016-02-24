#include <math.h>
#include <stdio.h>

#include "demosaic.h"

#define BLOCK 32

namespace gpu {
static texture<float, cudaTextureType2D, cudaReadModeElementType> gTex;
Demosaic *Demosaic::sInstance = 0x0;

void Demosaic::Initialize() 
{
  chan_desc = cudaCreateChannelDesc<float>();
}

Demosaic::~Demosaic()
{
  printf("Destroyed Demosaic\n");
}

__global__ void transform(float *dst, int w, int h, float a)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= w || y >= h)
    return;

  float u = x / (float)w;
  float v = y / (float)h;

  u -= 0.5f;
  v -= 0.5f;
  float tu = u * cosf(a) - v * sinf(a) + 0.5f;
  float tv = v * cosf(a) + u * sinf(a) + 0.5f;

  dst[y*w+x] = tex2D(gTex, tu, tv);
}

void Demosaic::Process(const float *src, float *dst, int w, int h, float a)
{
  cudaMallocArray(&cu_array, &chan_desc, w, h);
  cudaMemcpyToArray(cu_array, 0, 0, src, w*h*sizeof(float), cudaMemcpyHostToDevice);

  gTex.addressMode[0] = cudaAddressModeBorder; 
  gTex.addressMode[1] = cudaAddressModeBorder;
  gTex.filterMode = cudaFilterModeLinear;
  gTex.normalized = true;

  cudaBindTextureToArray(gTex, cu_array, chan_desc);
  float *output;
  cudaMalloc(&output, w*h*sizeof(float));
  dim3 dimBlock(BLOCK, BLOCK);
  dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x,
               (h + dimBlock.y - 1) / dimBlock.y);

  transform<<<dimGrid, dimBlock>>>(output, w, h, a);
  cudaMemcpy(dst, output, w*h*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFreeArray(cu_array);
  cudaFree(output);
}
} // namespace gpu
