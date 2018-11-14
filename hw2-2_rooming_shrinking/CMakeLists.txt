cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( hw2_2_rooming_shrinking )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)

add_executable( hw2_2_rooming_shrinking src/hw2_2_rooming_shrinking.cpp )
target_link_libraries( hw2_2_rooming_shrinking ${OpenCV_LIBS} )
