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

# location of source to build
if [ "$#" -gt 0 ]; then
    srcdir=$1
else
    wget glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    tar -xzf metis-5.1.0.tar.gz
    srcdir=${PWD}/metis-5.1.0
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
