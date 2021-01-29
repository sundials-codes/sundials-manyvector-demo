# Environment Setup and TPL Build Scripts

This directory contains scripts for setting up the environment on different
computing systems and installing a subset of the third-party libraries (TPLs)
necessary for running the demonstration code.

## Table of Contents

* [Environment Setup](#environment-setup)
* [Building Third-party Libraries](#building-third-party-libraries)

## Environment Setup

The following environment setup scripts are provided:

* `setup_summit.sh` - for the Summit supercomputer at ORNL
* `setup_lassen.sh` - for the Lassen supercomputer at LLNL
* `setup_quartz.sh` - for the Quartz supercomputer at LLNL

These scripts load various modules installed on the system (e.g., CMake, CUDA,
HDF5, etc.) and set several environment variables (e.g., `CXX`, `SUNDIALS_ROOT`,
`HDF5_ROOT`, etc.) referenced in the scripts for installing TPLs and read by
CMake when building the demonstration code to locate the installed TPLs.

To setup the default environment for a machine source the appropriate script
without any inputs i.e., `source ./setup_summit.sh` for Summit. Optional inputs
may be provided to alter the setup e.g., to use a non-default compiler or CUDA
version. For more details on the supported options see the comment block at
the top of each setup script as the inputs may vary depending on the machine.

## Building Third-party Libraries

The following scripts are provided to download, configure, build, and install
some of the TPLs needed by the demonstration code:

* `build-metis.sh` - install the Metis graph partitioning library
* `build-klu.sh` - install the KLU linear solver from the SuiteSparse library of
  sparse direct linear solvers (depends on Metis)
* `build-raja.sh` - install the RAJA performance portability library
* `build-sundials.sh` - install the SUNDIALS library of time integrators and
  nonlinear solvers (depends on KLU and RAJA)

For more details, see the comment block at the top of each build script.
