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
if [ "$#" -lt 3 ]; then
    echo "ERROR: Three (3) inputs required:"
    echo "  1) Path to metis source e.g., ~/metis-5.1.0"
    echo "  2) Metis version e.g., 5.1.0"
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
installdir=${PROJHOME}/${COMPILERNAME}/metis-${srcver}-${bldtype}

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
\rm -rf $installdir
mkdir -p $installdir

# configure
make config \
    CC=${CC} \
    CXX=${CXX} \
    prefix=$installdir \
    shared=1 \
    2>&1 | tee configure.log

# build and install
make install 2>&1 | tee install.log

# move log files
mv *.log $installdir/.
