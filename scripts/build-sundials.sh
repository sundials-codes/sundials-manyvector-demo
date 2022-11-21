#!/bin/bash -e
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

# clone SUNDIALS if necessary
if [ ! -d sundials ]; then
    git clone git@github.com:LLNL/sundials.git
    cd sundials
    git checkout v6.2.0
else
    cd sundials
fi

# remove old build dir
rm -rf build
mkdir build
cd build

# configure, make, and install
cmake \
    ../. \
    -D CMAKE_C_COMPILER=${CC} \
    -D CMAKE_C_FLAGS="-O3 -g" \
    -D CMAKE_CXX_COMPILER=${CXX} \
    -D CMAKE_CXX_FLAGS="-O3 -g" \
    -D SUNDIALS_INDEX_SIZE=32 \
    -D ENABLE_OPENMP=ON \
    -D ENABLE_MPI=ON \
    -D MPI_C_COMPILER=${MPICC} \
    -D MPI_CXX_COMPILER=${MPICXX} \
    -D MPI_Fortran_COMPILER=${MPIFC} \
    -D ENABLE_KLU=ON \
    -D KLU_INCLUDE_DIR=${KLU_ROOT}/include \
    -D KLU_LIBRARY_DIR=${KLU_ROOT}/lib \
    -D ENABLE_CUDA=ON \
    -D CMAKE_CUDA_ARCHITECTURES=70 \
    -D ENABLE_RAJA=ON \
    -D RAJA_DIR=${RAJA_ROOT} \
    -D SUNDIALS_RAJA_BACKENDS="CUDA" \
    -D camp_DIR=${CAMP_ROOT} \
    -D ENABLE_MAGMA=ON \
    -D MAGMA_DIR=${MAGMA_ROOT} \
    -D SUNDIALS_MAGMA_BACKENDS="CUDA" \
    -D CMAKE_INSTALL_PREFIX=${SUNDIALS_ROOT}

# make and install
make -j
make install
