cmake_minimum_required(VERSION 3.10)
project(least_square)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

find_package(Eigen3 REQUIRED)

# matplotlibcpp requires Python + matplotlib
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    include  # matplotlibcpp.h 위치
)


message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
# file(GLOB_RECURSE ALL_SRC "src/*.cpp")
# add_executable(${PROJECT_NAME} ALL_SRC)
add_executable(${PROJECT_NAME} src/least_square.cpp)

target_link_libraries(${PROJECT_NAME}
    ${Python3_LIBRARIES}
)