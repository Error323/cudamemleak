project (memleak CXX C)
cmake_minimum_required (VERSION 2.8)


#-------------------------------------------------------------------------------
# General Settings
#-------------------------------------------------------------------------------
set (CMAKE_CXX_FLAGS "-Wall -Wextra")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set (CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=native -mtune=native -ftree-vectorize -funsafe-math-optimizations -mfpu=neon-vfpv4")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 -gencode arch=compute_32,code=compute_32")
set (CUDA_NVCC_FLAGS_DEBUG "-G")
set (CUDA_NVCC_FLAGS_RELWITHDEBINFO "-G")
set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "Release")
endif (NOT CMAKE_BUILD_TYPE)

#-------------------------------------------------------------------------------
# Find packages
#-------------------------------------------------------------------------------
find_package (CUDA REQUIRED)
find_package (Glog REQUIRED)
find_package (GFlags REQUIRED)
find_package (OpenCV REQUIRED)


#-------------------------------------------------------------------------------
# Include sources
#-------------------------------------------------------------------------------
include (cmake/sources.cmake)
include_directories (
  ${OpenCV_INCLUDE_DIR} SYSTEM
  ${GFLAGS_INCLUDE_DIRS} SYSTEM
  ${GLOG_INCLUDE_DIRS} SYSTEM
)


#-------------------------------------------------------------------------------
# Define executable and link libraries
#-------------------------------------------------------------------------------
cuda_add_executable (${CMAKE_PROJECT_NAME} ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${OpenCV_LIBS}
)


#-------------------------------------------------------------------------------
# Status report
#-------------------------------------------------------------------------------
message (STATUS "")
message (STATUS " C/C++:")
message (STATUS "   C++ Compiler:                ${CMAKE_CXX_COMPILER}")
message (STATUS "   C++ flags (Release):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message (STATUS "   C++ flags (Debug):           ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message (STATUS "   C++ flags (RelWithDebInfo):  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message (STATUS " NVCC:")
message (STATUS "   NVCC Compiler:               ${CUDA_NVCC_EXECUTABLE}")
message (STATUS "   NVCC flags (Release):        ${CUDA_NVCC_FLAGS}")
message (STATUS "   NVCC flags (Debug):          ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_DEBUG}")
message (STATUS "   NVCC flags (RelWithDebInfo): ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELWITHDEBINFO}")
message (STATUS "")
message (STATUS " CMakeflags (${CMAKE_PROJECT_NAME}):")
message (STATUS "   CMAKE_BUILD_TYPE             ${CMAKE_BUILD_TYPE}")
message (STATUS "")
