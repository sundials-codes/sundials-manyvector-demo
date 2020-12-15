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
if [ "$#" -lt 3 ]; then
    echo "ERROR: Three (3) inputs required:"
    echo "  1) Path to SUNDIALS source"
    echo "  2) SUNDIALS version"
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
installdir=${PROJHOME}/${COMPILERNAME}/sundials-${srcver}-${bldtype}

# -------------------------------------------------------------------------------
# Setup Build
# -------------------------------------------------------------------------------

# return on any error
set -e

# build and install directories
builddir=${srcdir}/build

# remove old build directory, create new one, and move there
\rm -rf $builddir
mkdir -p $builddir
cd $builddir

# optimized or debug flags
if [ "$bldtype" == "opt" ]; then
    export CFLAGS='-O3'
    export CXXFLAGS='-O3'
    export FCFLAGS='-O3'
else
    export CFLAGS='-O0 -g -Wall'
    export CXXFLAGS='-O0 -g -Wall'
    export FCFLAGS='-O0 -g'
fi

# --------------------------------------------------------------------------
# Configure SUNDIALS
# --------------------------------------------------------------------------
cmake \
    -D BUILD_ARKODE=ON \
    -D BUILD_CVODE=ON \
    -D BUILD_CVODES=OFF \
    -D BUILD_IDA=OFF \
    -D BUILD_IDAS=OFF \
    -D BUILD_KINSOL=OFF \
    \
    -D SUNDIALS_PRECISION="double" \
    -D SUNDIALS_INDEX_SIZE="64" \
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
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_STATIC_LIBS=OFF \
    \
    -D CMAKE_INSTALL_PREFIX=$installdir \
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
make -j12 2>&1 | tee make.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\nmake returned $rc\n" | tee -a make.log
if [ $rc -ne 0 ]; then exit 1; fi

# --------------------------------------------------------------------------
# Install SUNDIALS
# --------------------------------------------------------------------------
make -j12 install 2>&1 | tee make.log

# check return code
rc=${PIPESTATUS[0]}
echo -e "\nmake install returned $rc\n" | tee -a install.log
if [ $rc -ne 0 ]; then exit 1; fi

# move log files
cp *.log $installdir/.

# done
exit 0
