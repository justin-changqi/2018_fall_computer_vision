cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( hw6-1_freq_filter_geo_tran )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include)

add_library ( cv_dft_2d lib/cv_dft_2d.cpp )
target_link_libraries( cv_dft_2d ${OpenCV_LIBS} )

add_library ( image_io lib/image_io.cpp )
target_link_libraries( image_io ${OpenCV_LIBS} )


add_executable( freq_filters_main src/freq_filters_main.cpp )
target_link_libraries( freq_filters_main ${OpenCV_LIBS} image_io cv_dft_2d)
