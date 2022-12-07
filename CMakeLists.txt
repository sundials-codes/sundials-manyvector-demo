# ------------------------------------------------------------------------------
# Programmer(s): David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# Copyright (c) 2020, Southern Methodist University and
# Lawrence Livermore National Security.
# All rights reserved.
# For details, see the LICENSE file.
# ------------------------------------------------------------------------------
# Top level CMakeLists.txt for the SUNDIALS demonstration application
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.0)

# prohibit in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source build prohibited.")
endif()

# default build type
set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")

project(SUNDIALS-DEMO C CXX)

# -------
# Macros
# -------

# Show/hide compiler and flags based on the current build type
macro(show_language_flags lang)
  set(_build_types DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
  foreach(build_type ${_build_types})
    if(${_cmake_build_type} MATCHES "${build_type}")
      message(STATUS "Appending ${lang} ${build_type} flags: ${CMAKE_${lang}_FLAGS_${build_type}}")
      mark_as_advanced(CLEAR CMAKE_${lang}_FLAGS_${build_type})
    else()
      mark_as_advanced(FORCE CMAKE_${lang}_FLAGS_${build_type})
    endif()
  endforeach()
  # show the language compiler and flags
  mark_as_advanced(CLEAR CMAKE_${lang}_COMPILER CMAKE_${lang}_FLAGS)
  # print implicit includes
  message(STATUS "Implicit ${lang} includes: ${CMAKE_${lang}_IMPLICIT_INCLUDE_DIRECTORIES}")
endmacro()

show_language_flags(C)
set(CMAKE_C_STANDARD "99" CACHE STRING "C standard to use")

show_language_flags(CXX)
set(CMAKE_CXX_STANDARD "11" CACHE STRING "C++ standard to use")

# Append to the runtime search path (rpath) of installed binaries any
# directories outside the project that are in the linker search path or contain
# linked library files.
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Set rpath" ON)
mark_as_advanced(FORCE CMAKE_INSTALL_RPATH_USE_LINK_PATH)

# --------
# Backend
# --------

set(RAJA_BACKEND "SERIAL"
  CACHE STRING "RAJA Backend to use: SERIAL, OPENMP, CUDA, HIP")
set_property(CACHE RAJA_BACKEND PROPERTY STRINGS "SERIAL;OPENMP;CUDA;HIP")

set(RAJA_ROOT "$ENV{RAJA_ROOT}"
  CACHE PATH "Root directory of RAJA install")

set(camp_DIR "$ENV{CAMP_ROOT}"
  CACHE PATH "Root directory of camp install")

find_package(RAJA REQUIRED
  PATHS
  "${RAJA_ROOT}/share/raja/cmake"
  "${RAJA_ROOT}")

find_package(Threads REQUIRED)

message(STATUS "Found RAJA: ${RAJA_DIR}")
mark_as_advanced(FORCE RAJA_DIR)

if(RAJA_BACKEND MATCHES "CUDA")
  enable_language(CUDA)
  show_language_flags(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "CUDA Architecture")
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
endif()

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
  "${SUNDIALS_ROOT}")

message(STATUS "Found SUNDIALS: ${SUNDIALS_DIR}")
mark_as_advanced(FORCE SUNDIALS_DIR)

# -----
# HDF5
# -----

option(ENABLE_HDF5 "Enable HDF5" OFF)

set(HDF5_ROOT "$ENV{HDF5_ROOT}"
  CACHE PATH "Root directory of HDF5 install")

if(ENABLE_HDF5)
  set(HDF5_PREFER_PARALLEL TRUE)
  set(HDF5_FIND_DEBUG TRUE)
  find_package(HDF5 REQUIRED COMPONENTS C HL)
endif()

# ---------------------
# add source directory
# ---------------------

add_subdirectory(src)

# -------------------
# install test files
# -------------------

# Install all text files, input data files, and scripts from the tests
# directory (and subdirectories) excluding any results files or directories.
install(DIRECTORY tests DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING
  PATTERN "*.txt"
  PATTERN "*.h5"
  PATTERN "*.lsf"
  PATTERN "*.sh"
  PATTERN "*.py"
  PATTERN "*results*" EXCLUDE)

# Install additional helpful scripts to the installed bin directory
install(FILES bin/cleanup.sh TYPE BIN)
install(FILES bin/utilities_euler3D.py TYPE BIN)
