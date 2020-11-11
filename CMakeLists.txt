cmake_minimum_required(VERSION 3.10)

project(main)

# CXX Configurations
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O2")

# Includes
include_directories(include)
file(GLOB Sources "src/*.cpp")
# Executables
add_executable(main main.cpp ${Sources})
