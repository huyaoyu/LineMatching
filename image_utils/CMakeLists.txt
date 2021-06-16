cmake_minimum_required(VERSION 3.16)
project(image_utils)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(OpenCV_DIR $ENV{OPENCV_ROOT_400}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED NO_DEFAULT_PATH)

set(LIB_NAME ${PROJECT_NAME})

add_library(${LIB_NAME} SHARED src/image_utils.cpp)
target_include_directories(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>)
target_link_libraries(${LIB_NAME}
        PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBRARIES})
