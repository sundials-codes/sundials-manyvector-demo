#!/bin/bash
# ------------------------------------------------------------------------------
# Setup Summit environment
# ------------------------------------------------------------------------------

# load modules
module load cmake
module load gcc/10.2.0
module load cuda/11.1.1
module load hdf5/1.10.7
module load raja/0.13.0
module load camp/0.1.0
module load magma/2.6.1
module load suite-sparse/5.9.0

# set environment variables
export CC=${OLCF_GCC_ROOT}/bin/gcc
export CXX=${OLCF_GCC_ROOT}/bin/g++
export FC=${OLCF_GCC_ROOT}/bin/gfortran

export MPICC=${MPI_ROOT}/bin/mpicc
export MPICXX=${MPI_ROOT}/bin/mpic++
export MPIFC=${MPI_ROOT}/bin/mpif90

export HDF5_ROOT=${OLCF_HDF5_ROOT}
export RAJA_ROOT=${OLCF_RAJA_ROOT}
export CAMP_ROOT=${OLCF_CAMP_ROOT}
export MAGMA_ROOT=${OLCF_MAGMA_ROOT}
export KLU_ROOT=${OLCF_SUITE_SPARSE_ROOT}
export SUNDIALS_ROOT="~/local/sundials-6.2.0"
