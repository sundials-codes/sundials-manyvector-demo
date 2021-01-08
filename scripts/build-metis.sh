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
# Script to build Metis
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

# ------------------------------------------------------------------------------
# Configure, build, and install
# ------------------------------------------------------------------------------

# return on any error
set -e

# move to source
cd $srcdir

# set realtype and indextypes to 64-bit
sed -i s/"#define REALTYPEWIDTH.*"/"#define REALTYPEWIDTH 64"/ include/metis.h
sed -i s/"#define IDXTYPEWIDTH.*"/"#define IDXTYPEWIDTH 64"/ include/metis.h

# set source and install directory paths
\rm -rf ${METIS_ROOT}
mkdir -p ${METIS_ROOT}

# configure
make config \
    CC=${CC} \
    CXX=${CXX} \
    prefix=${METIS_ROOT} \
    shared=1 \
    2>&1 | tee configure.log

# build and install
make install 2>&1 | tee install.log

# move log files
mv *.log ${METIS_ROOT}/.
