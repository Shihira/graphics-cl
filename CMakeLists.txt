cmake_minimum_required(VERSION 3.0)

project(gcl)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

add_subdirectory(gcl)
