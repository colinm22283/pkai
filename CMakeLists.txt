cmake_minimum_required(VERSION 3.24)
project(pkai LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)

#  -gencode arch=compute_61,code=sm_61
set(CMAKE_CUDA_FLAGS "-std=c++20 --extended-lambda -O3")

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

include_directories("${PROJECT_SOURCE_DIR}/header/")

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*cmake-build-debug.*")
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*cmake-build-release.*")

file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS "*.cu")
list(FILTER CU_SOURCES EXCLUDE REGEX ".*cmake-build-debug.*")
list(FILTER CU_SOURCES EXCLUDE REGEX ".*cmake-build-release.*")

add_subdirectory(lib)

function(add_project name sources)
    file(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/${name}/")

    add_executable(${name} ${sources})
    target_compile_options(${name} PRIVATE -O3 -flto)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/${name}/")
    target_link_libraries(${name} pkai ${CUDAToolkit_LIBRARIES})
endfunction()

add_subdirectory(projects)