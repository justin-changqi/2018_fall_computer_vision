cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11)
project( hw4-2_image_sharpning )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(include lib)

add_library (filters STATIC lib/filters.cpp)
target_link_libraries(filters ${OpenCV_LIBS})

add_executable( laplacian_sobel_filtering src/laplacian_sobel_filtering.cpp )
target_link_libraries( laplacian_sobel_filtering ${OpenCV_LIBS} filters)

add_executable( subtraction_operation src/subtraction_operation.cpp )
target_link_libraries( subtraction_operation ${OpenCV_LIBS} )
