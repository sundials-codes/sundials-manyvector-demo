# Environment, Build, and Post-processing Scripts

This directory contains scripts for setting up the environment on different
computing systems, building a subset of the libraries necessary for running
the demonstration code, and parsing the output from simulations.

The following environment setup scripts are provided:

* `setup_summit.sh` - for the Summit supercomputer at ORNL
* `setup_lassen.sh` - for the Lassen supercomputer at LLNL
* `setup_quartz.sh` - for the Quartz supercomputer at LLNL

These scripts load various modules installed on the system (e.g., CMake, CUDA,
HDF5, etc.) and set several environment variables (e.g., `CXX`, `SUNDIALS_ROOT`,
`HDF5_ROOT`, etc.) used in build scripts for installing third-party libraries
use in the demo code (see below). Additionally, these environment variables are
read by CMake when building the demo code to locate the installed third-party
libraries.

To setup the default environment for a machine source the appropriate script
without any inputs i.e., `source ./setup_summit.sh` for Summit. Optional inputs
may be provided to a script to alter the setup e.g., to use a non-default
compiler or CUDA version. For more details on the supported options see the
comment block at the top of the setup script.

The following build scripts are provided to configure and install some of the
libraries needed by the demo code:

* `build-metis.sh` - install the Metis graph partitioning library
* `build-klu.sh` - install the KLU linear solver from the SuiteSparse library of
  sparse direct linear solvers (depends on metis)
* `build-raja.sh` - install the RAJA performance portability library
* `build-sundials.sh` - install the SUNDIALS library of time integrators and
  nonlinear solvers (depends on KLU and RAJA)
