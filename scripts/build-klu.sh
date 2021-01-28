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
# Script to build KLU
# --------------------------------------------------------------------------

# location of source to build
if [ "$#" -gt 0 ]; then
    srcdir=$1
else
    wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.8.1.tar.gz
    tar -xzf SuiteSparse-5.8.1.tar.gz
    srcdir=${PWD}/SuiteSparse-5.8.1
fi

# build threads
bldthreads=12
if [ "$#" -gt 1 ]; then
    bldthreads=$2
fi

# ------------------------------------------------------------------------------
# Configure, build, and install
# ------------------------------------------------------------------------------

# return on any error
set -e

# move to source
cd $srcdir

# use external version of metis
\rm -rf metis-*

# clean any prior builds
make distclean

# disable packages we do not need by commenting out lines in Makefile
sed -i "/SPQR/s/^/#/g" Makefile
sed -i "/CXSparse/s/^/#/g" Makefile
sed -i "/CSparse/s/^/#/g" Makefile
sed -i "/UMFPACK/s/^/#/g" Makefile
sed -i "/SLIP_LU/s/^/#/g" Makefile
sed -i "/LDL/s/^/#/g" Makefile
sed -i "/RBio/s/^/#/g" Makefile
sed -i "/GraphBLAS/s/^/#/g" Makefile

# displays parameter settings; does not compile
make config \
    CC=${CC} \
    CXX=${CXX} \
    F77=${FC} \
    OPTIMIZATION="${OPTIMIZATION}" \
    BLAS=${BLAS_LIB} \
    LAPACK=${LAPACK_LIB} \
    MY_METIS_INC=${METIS_INC_DIR} \
    MY_METIS_LIB=${METIS_LIB} \
    GPU_CONFIG="" \
    INSTALL="${KLU_ROOT}" \
    JOBS=$bldthreads \
    2>&1 | tee configure.log

# compiles KLU
make library \
    CC=${CC} \
    CXX=${CXX} \
    F77=${FC} \
    OPTIMIZATION="${OPTIMIZATION}" \
    BLAS=${BLAS_LIB} \
    LAPACK=${LAPACK_LIB} \
    MY_METIS_INC=${METIS_INC_DIR} \
    MY_METIS_LIB=${METIS_LIB} \
    GPU_CONFIG="" \
    INSTALL="${KLU_ROOT}" \
    JOBS=$bldthreads \
    2>&1 | tee make.log

# create install directory
\rm -rf ${KLU_ROOT}
mkdir -p ${KLU_ROOT}

# install headers and libraries
cp -r include ${KLU_ROOT}
cp -r lib ${KLU_ROOT}

# move log files
cp *.log ${KLU_ROOT}/.
