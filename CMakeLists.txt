project (memleak CXX C)
cmake_minimum_required (VERSION 2.8)


#-------------------------------------------------------------------------------
# General Settings
#-------------------------------------------------------------------------------
set (CMAKE_CXX_FLAGS "-Wall -Wextra")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_20,code=compute_20")
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


#-------------------------------------------------------------------------------
# Include sources
#-------------------------------------------------------------------------------
include (cmake/sources.cmake)


#-------------------------------------------------------------------------------
# Define executable and link libraries
#-------------------------------------------------------------------------------
cuda_add_executable (${CMAKE_PROJECT_NAME} ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME})


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
