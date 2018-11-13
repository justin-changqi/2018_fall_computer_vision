cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( hw5_2d_dft_freq_filter )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g -O0")

include_directories(include)

add_library (his_eq STATIC lib/histogram_equalization.cpp)
target_link_libraries(his_eq ${OpenCV_LIBS})

add_library (dft2d STATIC lib/dft_2d.cpp)
target_link_libraries(dft2d ${OpenCV_LIBS} his_eq)

add_library (cvdft2d STATIC lib/cv_dft_2d.cpp)
target_link_libraries(cvdft2d ${OpenCV_LIBS})

add_executable( dft_2d src/dft_2d_main.cpp )
target_link_libraries( dft_2d ${OpenCV_LIBS} dft2d)

add_executable( cvdft_2d src/cv_dft_2d_main.cpp )
target_link_libraries( cvdft_2d ${OpenCV_LIBS} cvdft2d)

add_executable( dft_2d_filter src/dft_2d_filter.cpp )
target_link_libraries( dft_2d_filter ${OpenCV_LIBS} cvdft2d)

add_executable( noise_filter src/noise_filter_main.cpp )
target_link_libraries( noise_filter ${OpenCV_LIBS} cvdft2d)
