cmake_minimum_required(VERSION 3.5)

project(main)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
add_executable(main main.cpp cl_support.cpp)
target_link_libraries(main OpenCL::OpenCL ${OpenCV_LIBS})