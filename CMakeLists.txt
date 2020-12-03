cmake_minimum_required(VERSION 3.10)

project(main)

# CXX Configurations
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# options
option(OMP OFF)
option(CUDA OFF)
option(DEBUG OFF)

# Includes
include_directories(include)
file(GLOB Sources "src/*.cpp")


if(DEBUG)
    message("Debugging Turned On")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
elseif (PROFILE)
    message("Profileing code")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
else()
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()
if(OMP)
    message("Building with OpenMP")
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
if (CUDA)
    #set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.1/bin/nvcc")
    #set(CUDA_ROOT_DIR "/usr/local/cuda-10.1")
    #set(CUDA_BIN_PATH "/usr/local/cuda-10.1/bin/")
    #set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.1/samples/")
    find_package(CUDA REQUIRED)
    #find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #find_library(CUBLAS cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #find_library(CURAND curand ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #find_library(CUSOLVER cusolver ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #set(CMAKE_CUDA_STANDARD 11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_75,code=sm_75 -lcudart -lcublas -lcurand -lcusolver)
    #cuda_add_executable(src/cuda_math.cu
    #                    include/cuda_math.h)

    file(GLOB CudaSources "src/*.cu")
    file(GLOB CSources "src/*.c")
    #set(CMAKE_NVCC_FLAGS "-gencode arch=compute_75,coda=sm_75")
    include_directories(/usr/local/cuda-10.1/lib64)
    include_directories(/usr/local/cuda-10.1/samples/include)
    include_directories(/usr/local/cuda-10.1/samples/common/inc)
    include_directories(/usr/local/cuda-10.1/targets/x86_64-linux/include)
    #enable_language(CUDA)
    include(CheckLanguage)
    check_language(CUDA)
endif()
# Executables

if (CUDA)
    add_executable(main main.cpp ${Sources} ${CSources} ${CudaSources})
else()
    add_executable(main main.cpp ${Sources})
endif()
