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
#
# Usage: source ./setup_quartz.sh
#
# To see environment variables set by a module:
#   module show <module name>
#
# The default modules as of (7 Jan 2021):
#   1) intel/19.0.4
#   2) mvapich2/2.3
#   3) texlive/2016
#   4) StdEnv (S)
# --------------------------------------------------------------------------

# set compiler spec
compiler="intel@19.0.4"
if [ "$#" -gt 0 ]; then
    compiler=$1
fi

# get compiler name and version from spec
compilername="${compiler%%@*}"
compilerversion="${compiler##*@}"

# build type
bldtype="dbg"
if [ "$#" -gt 1 ]; then
    bldtype=$2
fi

if [[ "${bldtype}" != "dbg" && "${bldtype}" != "opt" ]]; then
    echo "ERROR: Unknown build type option: $bldtype"
    return 1
fi

# load compiler
case "$compilername" in
    intel)
        # default xl compiler as of 1 Sept 2019
        module load intel/${compilerversion}
        export CC=icc
        export CXX=icpc
        export FC=ifort
        ;;
    *)
        echo "ERROR: Unknown compiler option: $compiler"
        return 1
        ;;
esac

# export compiler name for use in build scripts
export COMPILERNAME="${compilername}-${compilerversion}"

# setup MPI compiler environment variables
export MPICC=$(which mpicc)
export MPICXX=$(which mpic++)
export MPIFC=$(which mpif90)

# optimization level (separate from C/CXX/F flags for KLU build)
if [ "$bldtype" == "opt" ]; then
    export OPTIMIZATION='-O3'
else
    export OPTIMIZATION='-O0'
fi

# other compiler flags
export CFLAGS='-g -Wall'
export CXXFLAGS='-g -Wall'
export FCFLAGS='-g'

# set environment variables
export HOST=quartz
export PROJHOME=${HOME}/local/${HOST}

# load other modules
module load cmake/3.18.0
module load mkl
module load hdf5-parallel

# list currently loaded modules
module list

# set environment variables
export BLAS_LIB=${MKLROOT}/lib/intel64/libmkl_rt.so
export LAPACK_LIB=${MKLROOT}/lib/intel64/libmkl_rt.so

export METIS_ROOT="${PROJHOME}/${COMPILERNAME}/metis-5.1.0-${bldtype}"
export METIS_INC_DIR=${METIS_ROOT}/include
export METIS_LIB=${METIS_ROOT}/lib/libmetis.so

export KLU_ROOT="${PROJHOME}/${COMPILERNAME}/suitesparse-5.8.1-${bldtype}"
export KLU_INC_DIR=${KLU_ROOT}/include
export KLU_LIB_DIR=${KLU_ROOT}/lib

export HDF5_ROOT=${HDF5}

export SUNDIALS_INDEX_SIZE=32
export SUNDIALS_CUDA_STATUS=OFF
export SUNDIALS_RAJA_STATUS=OFF
export SUNDIALS_ROOT="${PROJHOME}/${COMPILERNAME}/sundials-5.6.1-int${SUNDIALS_INDEX_SIZE}-${bldtype}"
