cmake_minimum_required(VERSION 3.14)
project(outlier_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -O3")
add_executable(outlier_detection example.cpp distance.hpp dataframe.hpp svm_cxx.hpp libsvm/svm.cpp libsvm/svm.h detection.hpp)