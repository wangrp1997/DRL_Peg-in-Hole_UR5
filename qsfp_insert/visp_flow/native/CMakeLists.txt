cmake_minimum_required(VERSION 3.16)
project(visp_flow_photometric LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(VISP REQUIRED COMPONENTS core visual_features vs)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(photometric_servo photometric_servo.cpp)
target_link_libraries(photometric_servo PRIVATE ${VISP_LIBRARIES})
target_include_directories(photometric_servo PRIVATE ${VISP_INCLUDE_DIRS})
