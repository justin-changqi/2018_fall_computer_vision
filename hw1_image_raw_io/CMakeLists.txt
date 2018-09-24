cmake_minimum_required(VERSION 2.8)
project( image_raw_io )
find_package( OpenCV REQUIRED )

# Homework 1.1 souce code
add_executable( image_raw_io_1 src/image_raw_io_1.cpp )
target_link_libraries( image_raw_io_1 ${OpenCV_LIBS} )

# Homework 1.2 souce code
add_executable( image_raw_io_2 src/image_raw_io_2.cpp )
target_link_libraries( image_raw_io_2 ${OpenCV_LIBS} )