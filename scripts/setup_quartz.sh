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
# Script to setup Quartz environment
# --------------------------------------------------------------------------
# Usage: source ./setup_quartz.sh
# --------------------------------------------------------------------------

# check for correct number of inputs
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please specify compiler: intel"
    return 1
fi

# set environment variables
export HOST=quartz
export PROJHOME=${HOME}/local/${HOST}

# load compiler
compiler=$1
case "$compiler" in
    intel)
        # default xl compiler as of 1 Sept 2019
        module load intel/19.0.4
        export COMPILERNAME="intel-19.0.4"
        export CC=icc
        export CXX=icpc
        export FC=ifort
        ;;
    *)
        echo "ERROR: Unknown compiler option: $compiler"
        return 1
        ;;
esac

# setup MPI compiler environment variables
export MPICC=mpicc
export MPICXX=mpic++
export MPIFC=mpif90

# load other modules
module load cmake
module load mkl
module load hdf5-parallel

# list currently loaded modules
module list

# set environment variables
export BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_rt.so"
export LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_rt.so"

METIS_ROOT="${PROJHOME}/${COMPILERNAME}/metis-5.1.0"
export METIS_INC_DIR="${METIS_ROOT}/include"
export METIS_LIB="${METIS_ROOT}/lib/libmetis.so"

KLU_ROOT="${PROJHOME}/${COMPILERNAME}/suitesparse-5.4.0"
export KLU_INC_DIR="${KLU_ROOT}/include"
export KLU_LIB_DIR="${KLU_ROOT}/lib"
