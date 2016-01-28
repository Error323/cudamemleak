#include <iostream>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

#include "init.cuh"
#include "process.cuh"

using namespace std;

int main(int argc, char *argv[])
{
  std::cout << "pid: " << getpid() << std::endl << std::endl;
  gpu::initCuda();
  vector<int> data(512,0);
  gpu::memOp(data.data(), data.size());
  return 0;
}
