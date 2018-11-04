# 2018_fall_computer_vision

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1551e93c45ad430c8befbfa452363eb8)](https://app.codacy.com/app/justin-changqi/2018_fall_computer_vision?utm_source=github.com&utm_medium=referral&utm_content=justin-changqi/2018_fall_computer_vision&utm_campaign=Badge_Grade_Dashboard)

Homework source code for 2018 fall term computer vision course at Taipei Tech MS program.

## System Environment
- Ubuntu16.04
- OpenCV 3.3
- CMake 2.8
- C++ 11

## Compile and execute code
This repository consists of mutiple CMake projects. Please follow the steps below for building the source code and execute.
```
# build
cd CMAKE_PROJECT_FOLDER
mkdir build && cd build
cmake ..
make
# execute
./EXE_FILE_NAME
```

## Crate new homework Cmake project
```
./create_new_hw.sh <project_name> <hw_order> <first_cpp_file> <second_cpp_file> ... 
```