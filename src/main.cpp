#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include "transform.h"

using namespace std;

int main(void)
{
  printf("pid: %i\n", getpid());

  int w = 64;
  int h = 64;
  float img[w*h];
  float res[w*h];

  for (int i = 0; i < w*h; i++)
    img[i] = 1.0f;

  int N = 1;
  for (int i = 0; i < N; i++)
    process(img, res, w, h, M_PI/4.0f);

  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
      printf("%i ", int(res[i*w+j]));
    printf("\n");
  }
  return 0;
}
