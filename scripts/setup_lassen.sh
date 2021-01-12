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
# Script to setup Lassen environment
# --------------------------------------------------------------------------
#
# Usage: source ./setup_lassen.sh
#
# To see environment variables set by a module:
#   module show <module name>
#
# The default modules as of (7 Jan 2021):
#   1) xl/2020.11.12
#   2) spectrum-mpi/rolling-release
#   3) cuda/10.1.243
#   4) StdEnv (S)
#
# hdf5-parallel/1.10.4 compilers (7 Jan 2021):
#   gcc/8.3.1
#   xl/2018.11.26
#   xl/2019.02.07
# --------------------------------------------------------------------------

# set compiler spec
compiler="xl@2019.02.07" # Default older xl compiler hdf5 was built with
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

# cuda version
cudaver="11.1.1"
if [ "$#" -gt 2 ]; then
    cudaver=$3
fi

# load compiler
case "$compilername" in
    xl)
        module load xl/${compilerversion}
        export CC=xlc_r
        export CXX=xlc++_r
        export FC=xlf_r
        ;;
    gcc)
        module load gcc/${compilerversion}
        export CC=gcc
        export CXX=g++
        export FC=gfortran
        ;;
    *)
        echo "ERROR: Unknown compiler option: $compiler"
        return 1
        ;;
esac

# export compiler name for use in build scripts
export COMPILERNAME="${compilername}-${compilerversion}"

# setup MPI compiler environment variables
export MPICC=${MPI_ROOT}/bin/mpicc
export MPICXX=${MPI_ROOT}/bin/mpic++
export MPIFC=${MPI_ROOT}/bin/mpif90

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
export HOST=lassen
export PROJHOME=${HOME}/local/${HOST}

# load other modules
module load cmake/3.18.0
module load essl/6.2.1
module load hdf5-parallel/1.10.4
module load cuda/${cudaver}
module load hpctoolkit/2019.03.10

# list currently loaded modules
module list

# set environment variables
export BLAS_LIB=${ESSLLIBDIR64}/libessl.so
export LAPACK_LIB=${ESSLLIBDIR64}/libessl.so

export METIS_ROOT="${PROJHOME}/${COMPILERNAME}/metis-5.1.0-${bldtype}"
export METIS_INC_DIR=${METIS_ROOT}/include
export METIS_LIB=${METIS_ROOT}/lib/libmetis.so

export KLU_ROOT="${PROJHOME}/${COMPILERNAME}/suitesparse-5.8.1-${bldtype}"
export KLU_INC_DIR=${KLU_ROOT}/include
export KLU_LIB_DIR=${KLU_ROOT}/lib

export HDF5_ROOT=${HDF5}

export RAJA_ROOT="${PROJHOME}/${COMPILERNAME}/raja-0.13.0-cuda-${cudaver}-${bldtype}"

export SUNDIALS_INDEX_SIZE=32
export SUNDIALS_ROOT="${PROJHOME}/${COMPILERNAME}/sundials-5.6.1-int${SUNDIALS_INDEX_SIZE}-cuda-${cudaver}-${bldtype}"
