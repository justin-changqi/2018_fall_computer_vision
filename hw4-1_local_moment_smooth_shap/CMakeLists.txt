cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( hw4-1_local_moment_smooth_shap )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)

add_executable( local_moment src/local_moment.cpp )
target_link_libraries( local_moment ${OpenCV_LIBS} )

add_executable( smoothing_mask src/smoothing_mask.cpp )
target_link_libraries( smoothing_mask ${OpenCV_LIBS} )
