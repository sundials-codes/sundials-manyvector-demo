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
# Script to build KLU on Summit
# --------------------------------------------------------------------------

# set paths
srcdir=${HOME}/SuiteSparse
installdir=${PROJWORK}/csc317/xl-16.1.1-3/suitesparse-5.4.0

# ------------------------------------------------------------------------------
# Configure, build, and install
# ------------------------------------------------------------------------------

# return on any error
set -e

# move to source
cd $srcdir

# use external version of metis
\rm -rf metis-*

# comment out all lines containing SPQR (i.e., don't build SPQR)
sed -i "/SPQR/s/^/#/g" Makefile

# displays parameter settings; does not compile
make config \
    CC=${CC} \
    CXX=${CXX} \
    F77=${FC} \
    BLAS="${OLCF_ESSL_ROOT}/lib64/libessl.so" \
    LAPACK="${OLCF_ESSL_ROOT}/lib64/libessl.so" \
    MY_METIS_INC="${OLCF_METIS_ROOT}/include" \
    MY_METIS_LIB="${OLCF_METIS_ROOT}/lib/libmetis.so" \
    INSTALL="$installdir" \
    JOBS=12 \
    2>&1 | tee suitesparse-5.4.0-configure.log

# compiles SuiteSparse
make library \
    CC=${CC} \
    CXX=${CXX} \
    F77=${FC} \
    BLAS="${OLCF_ESSL_ROOT}/lib64/libessl.so" \
    LAPACK="${OLCF_ESSL_ROOT}/lib64/libessl.so" \
    MY_METIS_INC="${OLCF_METIS_ROOT}/include" \
    MY_METIS_LIB="${OLCF_METIS_ROOT}/lib/libmetis.so" \
    INSTALL="$installdir" \
    JOBS=12 \
    2>&1 | tee suitesparse-5.4.0-make.log

# create install directory
\rm -rf $installdir
mkdir -p $installdir

# install headers and libraries
cp -r include $installdir
cp -r lib $installdir

# move log files
cp *.log $installdir/.
