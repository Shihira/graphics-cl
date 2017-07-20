#find_package(OpenCL REQUIRED)
find_package(SDL2 REQUIRED)

file(GLOB_RECURSE ALL_SOURCE src/*.cc src/common/*.cc) 

add_library(${PROJECT_NAME} STATIC ${ALL_SOURCE})
include_directories(${PROJECT_NAME}
    ${OpenCL_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIR})

file(GLOB_RECURSE ALL_UNIT_TESTS tests/*.cc)

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCL_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIR})

set(LINK_LIBS
    ${PROJECT_NAME}
    ${OpenCL_LIBRARIES}
    ${SDL2_LIBRARY})

set(OPTIMIZATION TRUE)

if(UNIX)
    add_definitions(-pipe)
    add_definitions(-Wall)
    add_definitions(-Wno-narrowing)
    add_definitions(-std=c++11)

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_definitions(-Wno-pessimizing-move)
        add_definitions(-Wno-missing-braces)

        #add_definitions(-fprofile-instr-generate -fcoverage-mapping)

        if(OPTIMIZATION)
            add_definitions(-O3)
            add_definitions(-fopenmp=libomp)
            set(LINK_LIBS ${LINK_LIBS} omp)
        endif()
    endif()
endif()

if(WIN32)
    add_definitions(/D _ITERATOR_DEBUG_LEVEL=0)
    if(OPTIMIZATION)
        add_definitions(/openmp)
        add_definitions(/O2 /Ob2 /Oi)
        add_definitions(/Gy /Qpar /arch:SSE2 /fp:fast)
        remove_definitions(/RTC1)
    endif()
endif()

foreach(TEST_FILE ${ALL_UNIT_TESTS})
    get_filename_component(EXEC_NAME ${TEST_FILE} NAME_WE)
    add_executable(${EXEC_NAME} ${TEST_FILE})
    include_directories(${EXEC_NAME} ${INCLUDE_DIRS})
    target_link_libraries(${EXEC_NAME} ${LINK_LIBS})
endforeach(TEST_FILE)

