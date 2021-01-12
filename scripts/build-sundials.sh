#!/bin/bash
# --------------------------------------------------------------------------
# Programmer(s): David J. Gardner @ LLNL
# --------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# --------------------------------------------------------------------------
# Script to build SUNDIALS
# --------------------------------------------------------------------------

# check for correct number of inputs
if [ "$#" -lt 1 ]; then
    echo "ERROR: Path to source required"
    exit 1
fi
srcdir=$1

# build threads
bldthreads=12
if [ "$#" -gt 1 ]; then
    bldthreads=$2
fi

# -------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------

# return on any error
set -e

# build and install directories
builddir=${srcdir}/build

# remove old build directory, create new one, and move there
\rm -rf $builddir
mkdir -p $builddir
cd $builddir

# --------------------------------------------------------------------------
# Configure SUNDIALS
# --------------------------------------------------------------------------
cmake \
    -D CMAKE_C_COMPILER=${CC} \
    -D CMAKE_C_FLAGS="${OPTIMIZATION} ${CFLAGS}" \
    -D CMAKE_CXX_COMPILER=${CXX} \
    -D CMAKE_CXX_FLAGS="${OPTIMIZATION} ${CXXFLAGS}" \
    \
    -D BUILD_ARKODE=ON \
    -D BUILD_CVODE=ON \
    -D BUILD_CVODES=OFF \
    -D BUILD_IDA=OFF \
    -D BUILD_IDAS=OFF \
    -D BUILD_KINSOL=OFF \
    \
    -D SUNDIALS_PRECISION="double" \
    -D SUNDIALS_INDEX_SIZE="$SUNDIALS_INDEX_SIZE" \
    \
    -D ENABLE_MPI=ON \
    -D MPI_C_COMPILER=${MPICC} \
    -D MPI_CXX_COMPILER=${MPICXX} \
    -D MPI_Fortran_COMPILER=${MPIFC} \
    \
    -D ENABLE_KLU=ON \
    -D KLU_INCLUDE_DIR="${KLU_INC_DIR}" \
    -D KLU_LIBRARY_DIR="${KLU_LIB_DIR}" \
    \
    -D ENABLE_CUDA=ON \
    -D CMAKE_CUDA_ARCHITECTURES="70" \
    \
    -D ENABLE_RAJA=ON \
    -D RAJA_DIR="${RAJA_ROOT}" \
    -D SUNDIALS_RAJA_BACKENDS="CUDA" \
    \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_STATIC_LIBS=OFF \
    \
    -D CMAKE_INSTALL_PREFIX=${SUNDIALS_ROOT} \
    \
    -D CMAKE_VERBOSE_MAKEFILE=OFF \
    \
    ../. | tee -a configure.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\ncmake returned $rc\n" | tee -a configure.log
if [ $rc -ne 0 ]; then exit 1; fi

# --------------------------------------------------------------------------
# Build SUNDIALS
# --------------------------------------------------------------------------
make -j $bldthreads 2>&1 | tee make.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\nmake returned $rc\n" | tee -a make.log
if [ $rc -ne 0 ]; then exit 1; fi

# --------------------------------------------------------------------------
# Install SUNDIALS
# --------------------------------------------------------------------------
make -j $bldthreads install 2>&1 | tee make.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\nmake install returned $rc\n" | tee -a install.log
if [ $rc -ne 0 ]; then exit 1; fi

# move log files
cp *.log ${SUNDIALS_ROOT}/.

# done
exit 0
