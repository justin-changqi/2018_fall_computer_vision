#!/bin/bash
_project_name=$1
_report_name=HW$2_106368002_report.tex
_src_codes_list=${@:3}
echo "Create project: "$_project_name
echo "Create source code: " 
mkdir -p $_project_name
cd $_project_name
mkdir -p build
mkdir -p images
mkdir -p report/img_src
mkdir -p result_img
mkdir -p src
rm -f CMakeLists.txt
touch CMakeLists.txt
cp ../report_temp.tex report/$_report_name
cp ../latex.gitignore report/.gitignore 
echo "cmake_minimum_required(VERSION 2.8)" >> CMakeLists.txt
echo "add_compile_options(-std=c++11)" >> CMakeLists.txt
echo "project( $_project_name )" >> CMakeLists.txt
echo "find_package( OpenCV REQUIRED )" >> CMakeLists.txt
echo "" >> CMakeLists.txt
echo "message(STATUS \"OpenCV library status:\")" >> CMakeLists.txt
echo "message(STATUS \"    config: \${OpenCV_DIR}\")" >> CMakeLists.txt
echo "message(STATUS \"    version: \${OpenCV_VERSION}\")" >> CMakeLists.txt
echo "message(STATUS \"    libraries: \${OpenCV_LIBS}\")" >> CMakeLists.txt
echo "message(STATUS \"    include path: \${OpenCV_INCLUDE_DIRS}\")" >> CMakeLists.txt
echo "" >> CMakeLists.txt
echo "include_directories(include)" >> CMakeLists.txt
echo "" >> CMakeLists.txt

cp -rf ../lib ./
cp -rf ../include ./

for filename in ./lib/*; 
do
    s=${filename##*/}
    s=${s##*/}
    libname=${s%.*}
    echo "$libname"
    echo "add_library ( "$libname lib/$s" )" >> CMakeLists.txt
    echo "target_link_libraries( "$libname "\${OpenCV_LIBS} )" >> CMakeLists.txt
    echo "" >> CMakeLists.txt
done

for file in $_src_codes_list
do
    echo "    ./src/$file"".cpp"
    cp -rf ../cpp_temp.cpp "src/$file"".cpp"
    cpp_file="src/$file"".cpp"
    # sed  -i '1s@^@#include "'"$file.hpp"'"@' $cpp_file
    echo "" >> CMakeLists.txt
    echo "add_executable( "$file src/$file.cpp" )" >> CMakeLists.txt
    echo "target_link_libraries( "$file "\${OpenCV_LIBS} image_io)" >> CMakeLists.txt
done
