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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()
if(OMP)
    message("Building with OpenMP")
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
if (CUDA)
    #find_package(CUDA REQUIRED)
endif()
# Executables
add_executable(main main.cpp ${Sources})
