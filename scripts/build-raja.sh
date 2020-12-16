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
# Script to build RAJA
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
# Configure
# --------------------------------------------------------------------------
cmake \
    -D CMAKE_C_COMPILER=${CC} \
    -D CMAKE_C_FLAGS="${OPTIMIZATION} ${CFLAGS}" \
    -D CMAKE_CXX_COMPILER=${CXX} \
    -D CMAKE_CXX_FLAGS="${OPTIMIZATION} ${CXXFLAGS}" \
    -D BLT_CXX_STD='c++14' \
    -D ENABLE_CUDA=ON \
    -D CMAKE_CUDA_STANDARD='14' \
    -D CUDA_ARCH='sm_70' \
    -D CMAKE_CUDA_ARCHITECTURES='70' \
    -D ENABLE_OPENMP=OFF \
    -D ENABLE_TARGET_OPENMP=OFF \
    -D RAJA_USE_DOUBLE=ON \
    -D CMAKE_INSTALL_PREFIX=${RAJA_ROOT} \
    -D CMAKE_VERBOSE_MAKEFILE=OFF \
    ../. | tee -a configure.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\ncmake returned $rc\n" | tee -a configure.log
if [ $rc -ne 0 ]; then exit 1; fi

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
make -j $bldthreads 2>&1 | tee make.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\nmake returned $rc\n" | tee -a make.log
if [ $rc -ne 0 ]; then exit 1; fi

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
make -j $bldthreads install 2>&1 | tee make.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\nmake install returned $rc\n" | tee -a install.log
if [ $rc -ne 0 ]; then exit 1; fi

# move log files
cp *.log ${RAJA_ROOT}/.

# done
exit 0
