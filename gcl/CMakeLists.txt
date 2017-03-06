find_package(OpenCL REQUIRED)
find_package(SDL2 REQUIRED)

file(GLOB_RECURSE ALL_SOURCE src/*.cc src/common/*.cc) 

add_library(${PROJECT_NAME} SHARED ${ALL_SOURCE})
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

foreach(TEST_FILE ${ALL_UNIT_TESTS})
    get_filename_component(EXEC_NAME ${TEST_FILE} NAME_WE)
    add_executable(${EXEC_NAME} ${TEST_FILE})
    include_directories(${EXEC_NAME} ${INCLUDE_DIRS})
    target_link_libraries(${EXEC_NAME} ${LINK_LIBS})
endforeach(TEST_FILE)

