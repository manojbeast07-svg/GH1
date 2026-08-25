cmake_minimum_required(VERSION 3.24)

# CMAKE_CUDA_ARCHITECTURES must be set before project()/enable_language(CUDA).
# "native" (CMake >= 3.24) builds for whatever GPU is present on this
# machine at configure time rather than hard-coding a compute capability.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(xray_cuda_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(cuda)
