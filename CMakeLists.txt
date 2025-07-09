cmake_minimum_required(VERSION 3.30)
project(leet_gpu CXX CUDA)
if (MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/utf-8\"")
endif()

add_definitions(-DLOCAL_MACHINE)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
file(GLOB_RECURSE src "challenges/*.cu" "challenges/*.cpp")
add_executable(leet_gpu ${src})
include_directories(include)

set_target_properties(leet_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(GTest CONFIG REQUIRED)
target_link_libraries(leet_gpu PRIVATE GTest::gtest_main CUDA::cudart)

include(GoogleTest)

enable_testing()
gtest_discover_tests(leet_gpu)