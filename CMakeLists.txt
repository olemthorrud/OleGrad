cmake_minimum_required(VERSION 3.20)
project(OleGrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(olegrad src/main.cpp)

# MNIST training executable
add_executable(mnist_train src/train_mnist.cpp)
