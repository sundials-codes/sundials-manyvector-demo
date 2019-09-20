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
# Script to setup Summit environment
# --------------------------------------------------------------------------
#
# Usage: source ./setup_summit.sh
#
# The default modules as of (1 Sept 2019):
#   1) xl/16.1.1-3
#   2) spectrum-mpi/10.3.0.1-20190611
#   3) hsi/5.0.2.p5
#   4) xalt/1.1.3
#   5) lsf-tools/2.0
#   6) darshan-runtime/3.1.7
#   7) DefApps
# To see environment variables set by a module:
#   module show <module name>
# --------------------------------------------------------------------------

# check for correct number of inputs
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please specify compiler xl or gcc"
    return 1
fi

# set environment variables
export HOST=summit
export PROJHOME=/ccs/proj/csc317

# load compiler
compiler=$1
case "$compiler" in
    xl)
        # default xl compiler as of 1 Sept 2019
        module load xl/16.1.1-3
        export COMPILERNAME="xl-16.1.1-3-r"
        export CC=${OLCF_XLC_ROOT}/bin/xlc_r
        export CXX=${OLCF_XLC_ROOT}/bin/xlc++_r
        export FC=${OLCF_XLF_ROOT}/bin/xlf_r
        ;;
    gcc)
        # default gcc compiler as of 1 Sept 2019
        module load gcc/6.4.0
        export COMPILERNAME="gcc-6.4.0"
        export CC=${OLCF_GCC_ROOT}/bin/gcc
        export CXX=${OLCF_GCC_ROOT}/bin/g++
        export FC=${OLCF_GCC_ROOT}/bin/gfortran
        ;;
    *)
        echo "ERROR: Unknown compiler option: $compiler"
        return 1
        ;;
esac

# setup MPI compiler environment variables
export MPICC=${MPI_ROOT}/bin/mpicc
export MPICXX=${MPI_ROOT}/bin/mpic++
export MPIFC=${MPI_ROOT}/bin/mpif90

# load other modules
module load cmake
module load essl
module load metis
module load hdf5
module load cuda
module load hpctoolkit

# unload modules (needed to build KLU)
module unload xalt

# list currently loaded modules
module list
