# ---------------------------------------------------------------
# Programmer: Daniel R. Reynolds @ SMU
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)

# set the project name and default languages
project(SUNDIALS-MANYVECTOR-MRISTEP-DEMONSTRATION C CXX)

# # build options
# option(ENABLE_CUDA "Enable CUDA" OFF)
# option(ENABLE_RAJA "Enable RAJA" OFF)
# option(ENABLE_OpenMP "Enable OpenMP" OFF)
# option(ENABLE_OpenMP_DEVICE "Enable OpenMP device offloading" OFF)

# create a root target
add_library(mpimanyvectordemo INTERFACE)
target_compile_features(mpimanyvectordemo INTERFACE cxx_std_11)
#target_include_directories(mpimanyvectordemo INTERFACE ${PROJECT_SOURCE_DIR}/include)

# check for MPI
find_package(MPI REQUIRED)
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# # check for CUDA
# if(ENABLE_CUDA)
#   if(NOT (DEFINED CMAKE_CUDA_HOST_COMPILER))
#       set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "NVCC host compiler")
#       mark_as_advanced(CMAKE_CUDA_HOST_COMPILER)
#   endif()
#   enable_language(CUDA)
#   message(STATUS "The CUDA host compiler is: ${CMAKE_CUDA_HOST_COMPILER}")
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
# endif()

# # check for OpenMP
# if(ENABLE_OpenMP)
#   find_package(OpenMP REQUIRED)
# endif()

# # check for OpenMP 4.5+
# if(ENABLE_OpenMP_DEVICE)
#   if(NOT OpenMP_FOUND)
#     find_package(OpenMP REQUIRED)
#   endif()
#   if(OpenMP_C_VERSION VERSION_LESS 4.5)
#     message(FATAL_ERROR "OpenMP >= 4.5 is required for device offloading!")
#   endif()
# endif()

# # check for RAJA
# if(ENABLE_RAJA)
#   find_package(RAJA CONFIG
#                PATHS ${RAJA_DIR} ${RAJA_DIR}/share/raja/cmake
#                REQUIRED)
# endif()

# check for SUNDIALS
include(cmake/SundialsConfig.cmake)

# set executable targets and dependencies
add_executable(communication_test_fluid.exe compile_test.cpp utilities.cpp io.cpp gopt.cpp)
target_link_libraries(communication_test_fluid.exe PRIVATE mpimanyvectordemo
                      SUNDIALS::nvecmpimanyvector
                      SUNDIALS::nvecparallel
                      SUNDIALS::nvecserial)
