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

# check for correct number of inputs
if [ "$#" -lt 3 ]; then
    echo "ERROR: Three (3) inputs required:"
    echo "  1) Path to SuiteSparse source e.g., ~/suitesparse-5.4.0"
    echo "  2) SuiteSparse version e.g., 5.4.0"
    echo "  3) Build type: opt or dbg"
    exit 1
fi

# path to SUNDIALS source and version name or number
srcdir=$1
srcver=$2

# build type: opt (optimized) or dbg (debug)
bldtype=$3
case "$bldtype" in
    opt|dbg) ;;
    *)
        echo "ERROR: Unknown build type: $bldtype"
        exit 1
        ;;
esac

# set install path
installdir=${PROJHOME}/${COMPILERNAME}/suitesparse-${srcver}-${bldtype}

# ------------------------------------------------------------------------------
# Configure, build, and install
# ------------------------------------------------------------------------------

# return on any error
set -e

# optimized or debug flags
if [ "$bldtype" == "opt" ]; then
    export FLAGS='-O3'
else
    export FLAGS='-O0 -g'
fi

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
    OPTIMIZATION="${FLAGS}" \
    BLAS=${BLAS_LIB} \
    LAPACK=${LAPACK_LIB} \
    MY_METIS_INC=${METIS_INC_DIR} \
    MY_METIS_LIB=${METIS_LIB} \
    INSTALL="$installdir" \
    JOBS=12 \
    2>&1 | tee configure.log

# compiles SuiteSparse
make library \
    CC=${CC} \
    CXX=${CXX} \
    F77=${FC} \
    OPTIMIZATION="${FLAGS}" \
    BLAS=${BLAS_LIB} \
    LAPACK=${LAPACK_LIB} \
    MY_METIS_INC=${METIS_INC_DIR} \
    MY_METIS_LIB=${METIS_LIB} \
    INSTALL="$installdir" \
    JOBS=12 \
    2>&1 | tee make.log

# create install directory
\rm -rf $installdir
mkdir -p $installdir

# install headers and libraries
cp -r include $installdir
cp -r lib $installdir

# move log files
cp *.log $installdir/.
