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

# set host
export HOST=summit

# load modules
module load cmake
module load essl
module load metis
module load hdf5
module load cuda

# unload modules (needed to build KLU)
module unload xalt

# set compiler environment variables
export CC=${OLCF_XLC_ROOT}/bin/xlc
export CXX=${OLCF_XLC_ROOT}/bin/xlc++
export FC=${OLCF_XLF_ROOT}/bin/xlf

export MPICC=${MPI_ROOT}/bin/mpicc
export MPICXX=${MPI_ROOT}/bin/mpic++
export MPIFC=${MPI_ROOT}/bin/mpif90
