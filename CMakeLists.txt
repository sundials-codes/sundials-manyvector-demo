# ------------------------------------------------------------------------------
# Programmer(s):  David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# Copyright (c) 2020, Southern Methodist University and
# Lawrence Livermore National Security.
# All rights reserved.
# For details, see the LICENSE file.
# ------------------------------------------------------------------------------
# Top level CMakeLists.txt for the SUNDIALS demonstration application
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.12)

project(SUNDIALS-DEMO C CXX)

# prohibit in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source build prohibited.")
endif()

# Show C and C++ compiler and flags. Hide build type specific flag options,
# except for the one corresponding to the current build type
set(_build_types DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
set(_languages C CXX)

foreach(lang ${_languages})
  foreach(build_type ${_build_types})
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
    if(${_cmake_build_type} MATCHES "${build_type}")
      message("Appending ${lang} ${build_type} flags")
      mark_as_advanced(CLEAR CMAKE_${lang}_FLAGS_${build_type})
    else()
      mark_as_advanced(FORCE CMAKE_${lang}_FLAGS_${build_type})
    endif()
  endforeach()
  # show the language compiler and flags
  mark_as_advanced(CLEAR CMAKE_${lang}_COMPILER CMAKE_${lang}_FLAGS)
endforeach()

# Set C and C++ standards
set(CMAKE_C_STANDARD "99" CACHE STRING "C standard to use")
set(CMAKE_CXX_STANDARD "11" CACHE STRING "C++ standard to use")

# ----
# MPI
# ----

find_package(MPI REQUIRED)

# ---------
# SUNDIALS
# ---------

set(SUNDIALS_ROOT "$ENV{SUNDIALS_ROOT}"
  CACHE PATH "Root directory of SUNDIALS install")

find_package(SUNDIALS REQUIRED
  PATHS
  "${SUNDIALS_ROOT}/lib64/cmake/sundials"
  "${SUNDIALS_ROOT}/lib/cmake/sundials"
  "${SUNDIALS_ROOT}"
  NO_DEFAULT_PATH)

mark_as_advanced(FORCE SUNDIALS_DIR)

# -----
# HDF5
# -----

option(ENABLE_HDF5 "Enable HDF5" OFF)

if(ENABLE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C HL)
endif()

# -----
# RAJA
# -----

option(ENABLE_RAJA "Enable RAJA" OFF)

set(RAJA_BACKEND "CUDA"
  CACHE STRING "RAJA backend to use")
set_property(CACHE RAJA_BACKEND PROPERTY STRINGS "CUDA;HIP")

set(RAJA_ROOT "$ENV{RAJA_ROOT}"
  CACHE PATH "Root directory of RAJA install")

if(ENABLE_RAJA)
  find_package(RAJA REQUIRED
    PATHS
    "${RAJA_ROOT}/share/raja/cmake"
    "${RAJA_ROOT}"
    NO_DEFAULT_PATH)
  # for some reason RAJA links to the Threads library
  find_package(Threads REQUIRED)
endif()

mark_as_advanced(FORCE RAJA_DIR)

# ---------------------
# add source directory
# ---------------------

add_subdirectory(src)
