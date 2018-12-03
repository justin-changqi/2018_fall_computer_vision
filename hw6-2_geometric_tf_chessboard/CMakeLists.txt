cmake_minimum_required(VERSION 3.0)
add_compile_options(-std=c++11)
project( hw6-2_geometric_tf_chessboard )
find_package( OpenCV REQUIRED )
find_package (Eigen3 REQUIRED NO_MODULE)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
  include
  # ${Eigen3_INCLUDE_DIRS}
)

add_library ( cv_dft_2d lib/cv_dft_2d.cpp )
target_link_libraries( cv_dft_2d ${OpenCV_LIBS} )

add_library ( image_io lib/image_io.cpp )
target_link_libraries( image_io ${OpenCV_LIBS} )

add_library ( geo_tf_chessboard lib/geo_tf_chessboard.cpp )
target_link_libraries( geo_tf_chessboard ${OpenCV_LIBS} )

add_executable( geometric_tf_chessboard src/geometric_tf_chessboard.cpp )
target_link_libraries( geometric_tf_chessboard 
                       ${OpenCV_LIBS} 
                       image_io 
                       geo_tf_chessboard)
