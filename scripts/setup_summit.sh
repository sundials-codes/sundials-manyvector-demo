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
# To see environment variables set by a module:
#   module show <module name>
#
# The default moduels as of 14 Dec 2020:
#   1) xl/16.1.1-5
#   2) spectrum-mpi/10.3.1.2-20200121
#   3) hsi/5.0.2.p5
#   4) xalt/1.2.1
#   5) lsf-tools/2.0
#   6) darshan-runtime/3.1.7
#   7) DefApps
# --------------------------------------------------------------------------

# set compiler spec
compiler="gcc@8.1.1"
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
        export CC=${OLCF_XLC_ROOT}/bin/xlc_r
        export CXX=${OLCF_XLC_ROOT}/bin/xlc++_r
        export FC=${OLCF_XLF_ROOT}/bin/xlf_r
        ;;
    gcc)
        module load gcc/${compilerversion}
        export CC=${OLCF_GCC_ROOT}/bin/gcc
        export CXX=${OLCF_GCC_ROOT}/bin/g++
        export FC=${OLCF_GCC_ROOT}/bin/gfortran
        ;;
    llvm)
        module load llvm/${complierversion}
        export CC=${OLCF_LLVM_ROOT}/bin/clang
        export CXX=${OLCF_LLVM_ROOT}/bin/clang++
        export FC=""
        ;;
    pgi)
        module load pgi/${compilerversion}
        export CC=${OLCF_PGI_ROOT}/linuxpower/${compilerversion}/bin/pgcc
        export CXX=${OLCF_PGI_ROOT}/linuxpower/${compilerversion}/bin/pgc++
        export FC=${OLCF_PGI_ROOT}/linuxpower/${compilerversion}/bin/pgfortran
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
export HOST=summit
export PROJHOME=/ccs/proj/csc317

# load other modules
module load cmake
module load essl
module load metis
module load hdf5
module load cuda/${cudaver}
#module load hip
module load hpctoolkit

# unload modules (needed to build KLU)
module unload xalt

# list currently loaded modules
module list

# set environment variables for library installs
export BLAS_LIB="${OLCF_ESSL_ROOT}/lib64/libessl.so"
export LAPACK_LIB="${OLCF_ESSL_ROOT}/lib64/libessl.so"

export METIS_INC_DIR="${OLCF_METIS_ROOT}/include"
export METIS_LIB="${OLCF_METIS_ROOT}/lib/libmetis.so"

export KLU_ROOT="${PROJHOME}/${COMPILERNAME}/suitesparse-5.8.1-${bldtype}"
export KLU_INC_DIR="${KLU_ROOT}/include"
export KLU_LIB_DIR="${KLU_ROOT}/lib"

export HDF5_ROOT="${OLCF_HDF5_ROOT}"

export RAJA_ROOT="${PROJHOME}/${COMPILERNAME}/raja-0.13.0-cuda-${cudaver}-${bldtype}"

export SUNDIALS_INDEX_SIZE=32
export SUNDIALS_ROOT="${PROJHOME}/${COMPILERNAME}/sundials-5.6.1-int${SUNDIALS_INDEX_SIZE}-cuda-${cudaver}-${bldtype}"

# user workspace for installing/running the demo
export SUNDIALS_DEMO_WORKSPACE=${MEMBERWORK}/csc317/sundials-demo
