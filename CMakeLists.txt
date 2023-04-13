cmake_minimum_required(VERSION 3.24)
project(pkai LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)

#  -g -G -O0
set(CMAKE_CUDA_FLAGS "-std=c++20 -O3 -gencode arch=compute_61,code=sm_61")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/header/")
include_directories("${PROJECT_SOURCE_DIR}/custom/header/")

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*cmake-build-debug.*")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*cmake-build-release.*")

file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS "*.cu")
list(FILTER CU_SOURCES EXCLUDE REGEX ".*cmake-build-debug.*")
list(FILTER CU_SOURCES EXCLUDE REGEX ".*cmake-build-release.*")

add_executable(pkai ${CPP_SOURCES} ${CU_SOURCES})

set_target_properties(pkai PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(pkai ${CUDA_LIBRARIES})