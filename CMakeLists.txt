cmake_minimum_required(VERSION 3.19)
project(LineMatching)

set(CMAKE_CXX_STANDARD 17)

# Boost.
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

FIND_PACKAGE(Eigen3 REQUIRED NO_MODULE)

# Ceres Solver.
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#set(OpenCV_DIR $ENV{OCV4_ROOT}/lib/cmake/opencv4)
set(OpenCV_DIR $ENV{OPENCV_ROOT_400}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED NO_DEFAULT_PATH)
message(STATUS "OpenCV_VERSION = ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_INSTALL_PATH = ${OpenCV_INSTALL_PATH}")
message(STATUS "OpenCV_LIBRARIES = ${OpenCV_LIBRARIES}")

include_directories(includes)

add_subdirectory(image_utils)

set(EXE_LINE_MATCHING LineMatching)
add_executable(${EXE_LINE_MATCHING} test_hough_transform.cpp)
target_link_libraries(${EXE_LINE_MATCHING} ${OpenCV_LIBRARIES} image_utils)

set(EXE_TEST_LSD test_lsd)
add_executable(${EXE_TEST_LSD} test_lsd.cpp)
target_link_libraries(${EXE_TEST_LSD} ${OpenCV_LIBRARIES} image_utils)

set(EXE_TEST_FLD test_fld)
add_executable(${EXE_TEST_FLD} test_fld.cpp)
target_link_libraries(${EXE_TEST_FLD} ${OpenCV_LIBRARIES} image_utils)

set(EXE_TEST_LSD_MATCHING test_lsd_matching)
add_executable(${EXE_TEST_LSD_MATCHING} test_lsd_matching.cpp)
target_link_libraries(${EXE_TEST_LSD_MATCHING} ${OpenCV_LIBRARIES} image_utils)

set(EXE_TEST_OPTIMIZATION test_optimization)
add_executable(${EXE_TEST_OPTIMIZATION} test_optimization.cpp)
target_link_libraries(${EXE_TEST_OPTIMIZATION}
        ${Boost_LIBRARIES}
        Eigen3::Eigen
        ${CERES_LIBRARIES}
        ${OpenCV_LIBRARIES}
        image_utils)