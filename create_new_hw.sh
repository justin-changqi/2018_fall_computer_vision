_project_name=$1
_report_name=HW#$2_106368002_report.tex
_src_codes_list=${@:3}
echo "Create project: "$_project_name
echo "Create source code: " 
mkdir -p $_project_name
cd $_project_name
mkdir -p build
mkdir -p images
mkdir -p include
mkdir -p report
mkdir -p result_img
mkdir -p src
rm CMakeLists.txt
touch CMakeLists.txt
touch report/$_report_name
echo "cmake_minimum_required(VERSION 2.8)" >> CMakeLists.txt
echo "add_compile_options(-std=c++11)" >> CMakeLists.txt
echo "project( "$_project_name" )" >> CMakeLists.txt
echo "find_package( OpenCV REQUIRED )" >> CMakeLists.txt
echo "" >> CMakeLists.txt
echo "message(STATUS \"OpenCV library status:\")" >> CMakeLists.txt
echo "message(STATUS \"    config: \${OpenCV_DIR}\")" >> CMakeLists.txt
echo "message(STATUS \"    version: \${OpenCV_VERSION}\")" >> CMakeLists.txt
echo "message(STATUS \"    libraries: \${OpenCV_LIBS}\")" >> CMakeLists.txt
echo "message(STATUS \"    include path: \${OpenCV_INCLUDE_DIRS}\")" >> CMakeLists.txt
echo "" >> CMakeLists.txt
echo "include_directories(include)" >> CMakeLists.txt

for file in $_src_codes_list
do
    echo "    ./include/$file"".hpp"
    echo "    ./src/$file"".cpp"
done
