cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( hw3_grey_tf_his_eqz )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)

add_executable( grey_level_transformation src/grey_level_transformation.cpp )
target_link_libraries( grey_level_transformation ${OpenCV_LIBS} )

add_executable( histogram_equalization src/histogram_equalization.cpp )
target_link_libraries( histogram_equalization ${OpenCV_LIBS} )
