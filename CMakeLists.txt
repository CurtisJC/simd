cmake_minimum_required(VERSION 3.12) # version can be different

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

add_compile_options(-O2 -std=c++23 -march=native)

project(simd) #name of your project

enable_testing() #to  discover tests in test explorer 

add_subdirectory(googletest) # add googletest subdirectory

include_directories(googletest/include) # this is so we can #include <gtest/gtest.h>

add_executable(simd_test simd_test.cpp) # add this executable

target_link_libraries(simd_test PRIVATE gtest) # link google test to this executable

include(GoogleTest)
gtest_discover_tests(simd_test) # discovers tests by asking the compiled test executable to enumerate its tests