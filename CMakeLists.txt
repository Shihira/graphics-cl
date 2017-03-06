cmake_minimum_required(VERSION 3.0)

project(gcl)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -Wall -Wno-narrowing -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -Wall -Wno-narrowing -std=c++11 -O0 -g")

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pessimizing-move -Wno-missing-braces")
    endif()
endif()

add_subdirectory(gcl)
