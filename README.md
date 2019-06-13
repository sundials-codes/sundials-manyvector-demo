# SUNDIALS Multirate+ManyVector Demo

[Note: this project is in active development; do not expect anything
to run (or even compile) at present.]

This is a SUNDIALS-based demonstration application to assess and
demonstrate the large-scale parallel performance of new capabilities
that have been added to SUNDIALS in recent years.  Namely:

1. ARKode's new multirate integration module, MRIStep, allowing
   high-order accurate calculations that subcycle "fast" processes
   within "slow" ones.

2. SUNDIALS' new MPIManyVector module, that allows extreme flexibility
   in how a solution "vector" is staged on computational resources.

To run this demo you will need modern C and C++ compilers.  All
dependencies (SUNDIALS and SuiteSparse) for the demo are installed
in-place using Spack, which is included in this repository.

Steps showing the process to download this demo code, install the
relevant dependencies, and build the demo in a Linux or OS X
environment are as follows:

    $ git clone git@bitbucket.org:drreynolds/sundials-manyvector-demo.git
    $ cd sundials-manyvector-demo
    $ .spack/bin/spack install sundials +suite-sparse +mpi
    $ .spack/bin/spack view symlink libs sundials
    $ .spack/bin/spack view symlink mpi mpi
    $ make

## Documentation
The documentation for this demo is in the folder Docs.  Please reference the 
files in this folder for details on the problem and solution approach.

## Authors
[Daniel R. Reynolds](mailto:reynolds@smu.edu)

