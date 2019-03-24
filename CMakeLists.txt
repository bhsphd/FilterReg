cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(poser)

#Build test or not
option(POSER_BUILD_TEST "Build the test programs or not. Need GoogleTest dependency." OFF)

#Add cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#CMake scripts
include(${CMAKE_SOURCE_DIR}/cmake/UtilFunctions.cmake)

#The compiler flags for c++ and cuda
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

#Cuda
find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
include_directories(${CUDA_INCLUDE_DIRS})

# Do not use the system Eigen
set(EIGEN_ROOT "${CMAKE_SOURCE_DIR}/external/eigen3")
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT})
include_directories(${EIGEN_INCLUDE_DIR})

#Point cloud library and a series
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

#Check the CUDA version and resolve confliction with pcl
if(${CUDA_VERSION_MAJOR} LESS 11)
    list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
    get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(vtk_flags)
    foreach(it ${dir_defs})
        if(it MATCHES "vtk*")
            list(APPEND vtk_flags ${it})
        endif()
    endforeach()

    foreach(d ${vtk_flags})
        remove_definitions(-D${d})
    endforeach()
endif()

#OpenCV lib: this version uses the customized build
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#The boost library is contained in pcl
include_directories(${Boost_INCLUDE_DIRS})

#The library for logging
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

#The google test library
if(POSER_BUILD_TEST)
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
endif(POSER_BUILD_TEST)

#The project directory shall also be included
include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/external")

#The submodule
add_subdirectory(common)
add_subdirectory(geometry_utils)
add_subdirectory(visualizer)
add_subdirectory(imgproc)
add_subdirectory(cloudproc)
add_subdirectory(corr_search)
add_subdirectory(kinematic)
add_subdirectory(ransac)

#The applications
add_subdirectory(apps)
