cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( grey_level_resolution )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)

# Homework 2.1 souce code
add_executable( hw2_1_grey_level_resolution src/hw2_1_grey_level_resolution.cpp )
target_link_libraries( hw2_1_grey_level_resolution ${OpenCV_LIBS} )