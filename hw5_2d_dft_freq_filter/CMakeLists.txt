cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( hw5_2d_dft_freq_filter )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)

add_executable( dft_2d src/dft_2d.cpp )
target_link_libraries( dft_2d ${OpenCV_LIBS} )

add_executable( noise_filter src/noise_filter.cpp )
target_link_libraries( noise_filter ${OpenCV_LIBS} )
