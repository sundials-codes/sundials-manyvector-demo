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
# Usage: source ./setup_lassen.sh
#
# The default modules as of (25 Nov 2019):
#   1) xl/2019.08.20
#   2) spectrum-mpi/rolling-release
#   3) cuda/10.1.243
#   4) StdEnv
# To see environment variables set by a module:
#   module show <module name>
# --------------------------------------------------------------------------

# check for correct number of inputs
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please specify compiler: xl"
    return 1
fi

# set environment variables
export HOST=lassen
export PROJHOME=${HOME}/local/${HOST}

# load compiler
compiler=$1
case "$compiler" in
    xl)
        # use older xl compiler that hdf5 was built with
        module load xl/2019.02.07
        export COMPILERNAME="xl-2019.08.20-r"
        export CC=xlc_r
        export CXX=xlc++_r
        export FC=xlf_r
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

# enable building Sundials with CUDA support
export SUNDIALS_CUDA="ON"

# load other modules
module load cmake
module load essl/6.2
module load hdf5-parallel/1.10.4
module load cuda/10.1.243
module load hpctoolkit/2019.03.10

# list currently loaded modules
module list

# set environment variables
ESSL_ROOT=/usr/tcetmp/packages/essl/essl-6.2
export BLAS_LIB=${ESSL_ROOT}/lib64/libessl.so
export LAPACK_LIB=${ESSL_ROOT}/lib64/libessl.so

METIS_ROOT=${PROJHOME}/${COMPILERNAME}/metis-5.1.0
export METIS_INC_DIR=${METIS_ROOT}/include
export METIS_LIB=${METIS_ROOT}/lib/libmetis.so

KLU_ROOT=${PROJHOME}/${COMPILERNAME}/suitesparse-5.4.0
export KLU_INC_DIR=${KLU_ROOT}/include
export KLU_LIB_DIR=${KLU_ROOT}/lib
