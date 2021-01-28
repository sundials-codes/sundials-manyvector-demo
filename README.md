# SUNDIALS ManyVector+Multirate Demonstration Code

[Note: this project is in active development.]

This is a SUNDIALS-based demonstration application to assess and demonstrate the
large-scale parallel performance of new capabilities that have been added to
SUNDIALS in recent years. Namely:

1. The new SUNDIALS MPIManyVector module, that allows extreme flexibility in how
   a solution "vector" is staged on computational resources (e.g., CPUs and
   GPUs).

2. The new ARKODE multirate integration module, MRIStep, allowing high-order
   accurate calculations that subcycle "fast" processes within "slow" ones.

3. The new flexible SUNDIALS linear solver interfaces, to enable streamlined use
   of problem specific and scalable linear solver libraries (e.g., *hypre*,
   PETSc and Trilinos).

## Table of Contents

* [Model Equations](#model-equations)
* [Discretization](#discretization)
* [Building](#building)
* [Running](#running)
* [Adding New Tests](#adding-new-tests)
* [Authors](#authors)

## Model Equations

This code simulates a 3D nonlinear inviscid compressible Euler
equation with advection and reaction of chemical species,
<p align="center"><img src="/tex/77d55513ca92983b2a16bf2c0d217d53.svg?invert_in_darkmode&sanitize=true" align=middle width=208.23551595pt height=16.438356pt/></p>
for independent variables
<p align="center"><img src="/tex/b1fb667c457471aa9a6fe9b8a16c5c51.svg?invert_in_darkmode&sanitize=true" align=middle width=227.07922545pt height=17.031940199999998pt/></p>
where the spatial domain is a three-dimensional cube,
<p align="center"><img src="/tex/724344359d58ffb008954bfadd390bc2.svg?invert_in_darkmode&sanitize=true" align=middle width=210.46034459999998pt height=16.438356pt/></p>
The differential equation is completed using initial condition
<p align="center"><img src="/tex/5270774dab046199071bafaa705bfac0.svg?invert_in_darkmode&sanitize=true" align=middle width=128.36284064999998pt height=16.438356pt/></p>
and face-specific boundary conditions, [xlbc, xrbc] x [ylbc, yrbc] x
[zlbc, zrbc], where each may be any one of

* periodic (0),
* homogeneous Neumann (1),
* homogeneous Dirichlet (2), or
* reflecting (3)

under the restriction that if any boundary is set to "periodic" then
the opposite face must also indicate a periodic condition.

Here, the 'solution' is given by
<img src="/tex/28187201df21690b7d306712905d18e4.svg?invert_in_darkmode&sanitize=true" align=middle width=468.60549779999997pt height=35.5436301pt/>,
that corresponds to the density, x,y,z-momentum, total energy
per unit volume, and any number of chemical densities
<img src="/tex/b8b6bd2b662b75b8e135f6da0bd323ce.svg?invert_in_darkmode&sanitize=true" align=middle width=79.96356884999999pt height=27.91243950000002pt/> that are advected along with the
fluid.  The fluxes are given by
<p align="center"><img src="/tex/3a03e5f8f13d64219de34f4e89ece974.svg?invert_in_darkmode&sanitize=true" align=middle width=425.4109365pt height=23.5253469pt/></p>
<p align="center"><img src="/tex/6de6c35f4b2af691b7c615d7455c9eb9.svg?invert_in_darkmode&sanitize=true" align=middle width=423.2250824999999pt height=23.9085792pt/></p>
<p align="center"><img src="/tex/9ab5c11c75414d0f6469ee8b93a5d46b.svg?invert_in_darkmode&sanitize=true" align=middle width=429.71667914999995pt height=23.5253469pt/></p>
The external force 
<p align="center"><img src="/tex/fdd8b43797ed57d5386562a0b6c3a124.svg?invert_in_darkmode&sanitize=true" align=middle width=72.46420829999998pt height=24.65753399999998pt/></p>
is test-problem-dependent, and the ideal gas equation of state gives
<p align="center"><img src="/tex/272a02648ae7ce4db259539aa98655dc.svg?invert_in_darkmode&sanitize=true" align=middle width=218.4856146pt height=36.09514755pt/></p>
and
<p align="center"><img src="/tex/fcab1314e35d6d578ef90227b587c829.svg?invert_in_darkmode&sanitize=true" align=middle width=200.67741375pt height=29.47417935pt/></p>
or equivalently,
<p align="center"><img src="/tex/c525acb8ade640e26804cc23100fc642.svg?invert_in_darkmode&sanitize=true" align=middle width=250.13614230000002pt height=30.1801401pt/></p>
and
<p align="center"><img src="/tex/bb93cc09da34e3117795a4e98a4e4db6.svg?invert_in_darkmode&sanitize=true" align=middle width=215.21714114999997pt height=32.6705313pt/></p>

We have the physical parameters:

* R is the specific ideal gas constant (287.14 J/kg/K).

* <img src="/tex/aa8cfea83e4502fbd685d6c095494147.svg?invert_in_darkmode&sanitize=true" align=middle width=14.102064899999991pt height=14.15524440000002pt/> is
the specific heat capacity at constant volume (717.5 J/kg/K),

* <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> is the ratio of specific heats, <img src="/tex/f8415659af3e4a9e110591f46cc2875e.svg?invert_in_darkmode&sanitize=true" align=middle width=113.34245999999997pt height=28.670654099999997pt/> (1.4),

corresponding to air (predominantly an ideal diatomic gas). The speed
of sound in the gas is then given by
<p align="center"><img src="/tex/e55dd025376e04a1ada428d642da3089.svg?invert_in_darkmode&sanitize=true" align=middle width=67.10942039999999pt height=39.452455349999994pt/></p>
The fluid variables above are non-dimensionalized; in standard SI
units these would be:

* [rho] = kg / m<img src="/tex/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>,

* [vx] = [vy] = [vz] = m/s, which implies that [mx] = [my] = [mz] = kg / m<img src="/tex/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/> / s, and

* [et] = kg / m / s<img src="/tex/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>

* [c_i] = kg / m<img src="/tex/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>

Note: the fluid portion of the above description follows section 7.3.1-7.3.3 of
https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html

## Discretization

This program solves the above problem using a finite volume spatial
semi-discretization over a uniform grid of dimensions `nx` x `ny` x `nz`, with
fluxes calculated using a 5th-order FD-WENO reconstruction. The spatial domain
uses a 3D domain decomposition approach for parallelism over `nprocs` MPI
processes, with layout `npx` x `npy` x `npz` defined automatically via the
`MPI_Dims_create` utility routine.  The minimum size for any dimension is 3, so
to run a two-dimensional test in the yz-plane, one could specify `nx = 3` and
`ny = nz = 200` when run in parallel, only 'active' spatial dimensions (those
with extent greater than 3) will be parallelized.

Each fluid field
(<img src="/tex/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode&sanitize=true" align=middle width=8.49888434999999pt height=14.15524440000002pt/>, <img src="/tex/f8eec81a1374c2e08228fb574a0e5fdf.svg?invert_in_darkmode&sanitize=true" align=middle width=21.88747274999999pt height=14.15524440000002pt/>, <img src="/tex/e4c6c96061743e44c44edafd6e06abe7.svg?invert_in_darkmode&sanitize=true" align=middle width=21.512706599999987pt height=14.15524440000002pt/>, <img src="/tex/b9034568c7237b47ca94b79611bd9fd9.svg?invert_in_darkmode&sanitize=true" align=middle width=21.18545879999999pt height=14.15524440000002pt/> and <img src="/tex/71c0437a67c94e48f18cc11d0c17a38c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61992929999999pt height=14.15524440000002pt/>)
is stored in its own parallel `N_Vector` object. Chemical species at all spatial
locations over a single MPI rank are collocated into a single serial or RAJA
`N_Vector` object when running on the CPU or GPU respectively. The five fluid
vectors and the chemical species vector are combined together to form the full
"solution" vector
 <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/>
using the `MPIManyVector` `N_Vector` module.

For non-reactive flows, the resulting initial-value problem is evolved in times
using an adaptive step explicit Runge-Kutta method from the ARKStep module in
ARKODE. For problems involving (typically stiff) chemical reactions, the
multirate initial-value problem is solved using the MRIStep module in ARKODE,
wherein the gas dynamics equations are evolved explicitly at the 'slow' time
scale, while the chemical kinetics are evolved using a temporally-adaptive,
diagonally-implicit Runge-Kutta method from the ARKStep module.  The MPI
rank-local implicit systems are solved using the default (modified or inexact)
Newton nonlinear solver, with a custom linear solver that solves each rank-local
linear system using either the KLU, cuSPRASE batched-QR, or GMRES
SUNLinaerSolver linear solver module.

## Building

Steps showing the process to download this demonstration code, install the
relevant dependencies, and build the code in a Linux or OS X environment are as
follows. To obtain the demonstration code simply clone this repository with Git:

```bash
  git clone https://github.com/sundials-codes/sundials-manyvector-demo.git
```

To compile the code you will need:

* modern C and C++ compilers

* [CMake](https://cmake.org) 3.12 or newer

* an MPI library e.g., [OpenMPI](https://www.open-mpi.org/),
  [MPICH](https://www.mpich.org/), etc.

* the [SUNDIALS](https://computing.llnl.gov/projects/sundials) library of time
  integrators and nonlinear solvers

* the [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) library
  of sparse direct linear solvers (specifically KLU)

* the [HDF5](https://www.hdfgroup.org/) high-performance data management and
  storage suite

For running on systems with GPUs you will additionally need:

* the NVIDIA [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (the nvcc
  compiler and cuSPRASE library)

* the [RAJA](https://github.com/LLNL/RAJA) performance portability library

To assist in building the code the [scripts](./scripts) directory contains shell
to setup the environment on specific systems and install some of the required
dependencies. For example, if working on Summit the following commands may be
used to setup the environment and install the necessary dependencies:

```bash
  cd sundials-manyvector-demo/scripts
  export PROJHOME=/css/proj/[projid]
  source setup_summit.sh
  ./build-klu.sh
  ./build-raja.sh
  ./build-sundials.sh
```

Where `[projid]` is the Summit project ID. For more information on the setup and
build scripts see the README file in the [scripts](./scripts) directory. As an
alternative, any of the dependencies for the demonstration code can be installed
with the [Spack](https://github.com/spack/spack) package manager e.g.,

```bash
  git clone https://github.com/spack/spack.git
  spack/bin/spack install mpi
  spack/bin/spack install hdf5 +mpi +pic +szip
  spack/bin/spack isntall suitesparse
  spack/bin/spack install raja +cuda
  spack/bin/spack install sundials +klu +mpi +raja +cuda
```

Once the necessary dependencies are installed, the following CMake variables can
be used to configure the demonstration code build:

* `CMAKE_INSTALL_PREFIX` - the path where executables and input files should be
  installed e.g., `path/to/myinstall`. The executables will be installed in the
  `bin` directory and input files in the `tests` directory under the given path.

* `CMAKE_C_COMPILER` - the C compiler to use e.g., `mpicc`

* `CMAKE_C_FLAGS` - the C compiler flags to use e.g., `-g -O2`

* `CMAKE_C_STANDARD` - the C standard to use, defaults to `99`

* `CMAKE_CXX_COMPILER` - the C++ compiler to use e.g., `mpicxx`

* `CMAKE_CXX_FLAGS` - the C++ flags to use e.g., `-g -O2`

* `CMAKE_CXX_STANDARD` - the C++ standard to use, defaults to `11`

* `SUNDIALS_ROOT` - the root directory of the SUNDIALS installation, defaults to
  the value of the `SUNDIALS_ROOT` environment variable

* `ENABLE_RAJA` - build with RAJA support, defaults to `OFF`

* `RAJA_ROOT` - the root directory of the RAJA installation, defaults to the
  value of the `RAJA_ROOT` environment variable

* `RAJA_BACKEND` - set the RAJA backend to use with the demonstration code,
  defaults to `CUDA`

* `ENABLE_HDF5` - build with HDF5 I/O support, defaults to `OFF`

* `HDF5_ROOT` - the root directory of the HDF5 installation, defaults to the
  value of the `HDF5_ROOT` environment variable

When RAJA is enabled with the CUDA backend the following additional variables
may also be set:

* `CMAKE_CUDA_COMPILER` - the CUDA compiler to use e.g., `nvcc`

* `CMAKE_CUDA_FLAGS` - the CUDA compiler flags to use

* `CMAKE_CUDA_ARCHITECTURES` - the CUDA architecture to target, defaults to `70`

In-source builds are not permitted and as such the code should be configured and
built from a separate build directory. For example, continuing with the Summit
case from above, the following commands can be used to build with RAJA targeting
CUDA and HDF5 output enabled:

```bash
  cd sundials-manyvector-demo
  mkdir build
  cd build
  cmake ../. \
    -DCMAKE_INSTALL_PREFIX="${MEMBERWORK}/[projid]/sundials-demo" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_C_FLAGS="-g -O2" \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-g -O2" \
    -DENABLE_RAJA="ON" \
    -DENABLE_HDF5="ON"
  make
  make install
```

The test executables and input files are installed in the member work space for
the Summit project ID given by `[projid]`.

**Note:** In this example, since the environment was configured using the Summit
setup script, the values for `SUNDIALS_ROOT`, `RAJA_ROOT`, and `HDF5_ROOT` can
be omitted from the `cmake` command as these values are automatically set from
the corresponding environment variables defined by the setup script.

## Running

Several test cases are included with the code and the necessary input files for
each case are contained in the subdirectories within the [tests](./tests)
directory. Each input file is internally documented to discuss all possible
input parameters (in case some have been added since this `README` was last
updated).

The input files contain parameters to set up the physical problem:

* spatial domain, <img src="/tex/9432d83304c1eb0dcb05f092d30a767f.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/> -- `xl`, `xr`, `yl`, `yr`, `zl`, `zr`

* time interval, <img src="/tex/bbde6652efaeb60e967ee67be6440eb7.svg?invert_in_darkmode&sanitize=true" align=middle width=46.033257599999985pt height=24.65753399999998pt/> -- `t0`, `tf`

* the ratio of specific heats, <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> -- `gamma`

* spatial discretization dimensions -- `nx`, `ny`, `nz`

* boundary condition types -- `xlbc`, `xrbc`, `ylbc`, `yrbc`, `zlbc`, `zrbc`

Parameters to control the execution of the code:

* desired cfl fraction -- `cfl` (if set to zero, then the time step is
 chosen purely using temporal adaptivity).

* number of desired solution outputs -- `nout`

* a flag to enable optional output of RMS averages for each field at
 the frequency spefied via `nout` -- `showstats`

Numerous parameters are also provided to control how time integration is
performed (these are passed directly to ARKODE). For further information on the
ARKODE solver parameters and the meaning of individual values, see the
[ARKODE documentation](http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/index.html).

To specify an input file to the executable, the input filename should be
provided using the `-f` flag e.g.,

```bash
  <executable> -f <input_file>
```

Additionally, any input parameters may also be specified on the
command line e.g.,

```bash
  <executable> --nx=100 --ny=100 --nz=400
```

For example, continuing with the Summit case from above, the primordial blast
test can be run on one Summit node using four cores and four GPUs with the
following commands:

```bash
  cd ${MEMBERWORK}/[projid]/sundials-demo/tests/primordial_blast
  bsub -q debug -nnodes 1 -W 0:10 -P [projid] -Is $SHELL
  jsrun -n4 -a1 -c1 -g1 ../../bin/primordial_blast_mr.exe -f input_primordial_blast_mr_gpu.txt
```

The `bsub` command above will submit a request for an interactive job to the
debug queue allocating one node for 10 minutes with the compute time charged to
`[projid]`. Once the interactive session starts the test case is launched using
the `jsrun` command. Solutions are output to disk using parallel HDF5, solution
statistics are optionally output to the screen at specified frequencies, and run
statistics are printed at the end of the simulation.

The parallel HDF5 to solution snapshots are written at the frequency specified
by `nout`.  Accompanying these `output-#######.hdf5` files is an automatically
generated input file, `restart_parameters.txt` that stores a complete set of
input parameters to restart the simulation from the most recently generated
output file. This is a "warm" restart, in that it will pick up the calculation
where the previous one left off, using the same initial time step size as
ARKStep would use. This restart may differ slightly from an uninterrupted run
since other internal ARKStep time adaptivity parameters cannot be reused.  We
note that the restart must use the same spatial grid size and number of chemical
tracers as the original run, but it may use a different number of MPI tasks if
desired.

## Adding New Tests

Individual test problems are uniquely specified through an input file and
auxiliary source code file(s) that should be linked with the main routine at
compile time. By default, all codes are built with no chemical species; however,
this may be controlled at compilation time using the `NVAR` preprocessor
directive, corresponding to the number of unknowns at any spatial location.
Hence, the (default) minimum value for `NVAR` is 5, so for a calculation with 4
chemical species the code should be compiled with the preprocessor directive
`NVAR=9`. See [src/CMakeLists.txt](./src/CMakeLists.txt) for examples of how to
specify `NVAR` when adding a new test/executable.

The auxiliary source code files for creating a new test must contain three
functions. Each of these must return an integer flag indicating success (0) or
failure (nonzero). The initial condition function
<img src="/tex/d3cb4393199b89ca003e78d3486fa147.svg?invert_in_darkmode&sanitize=true" align=middle width=46.837068299999984pt height=24.65753399999998pt/>
must have the signature

```C++
  int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);
```

and the forcing function
<img src="/tex/c441e18e502be64ac772003edac839dc.svg?invert_in_darkmode&sanitize=true" align=middle width=52.94748029999999pt height=24.65753399999998pt/>
must have the signature

```C++
  int external_forces(const realtype& t, N_Vector G, const UserData& udata);
```

Additionally, a function must be supplied to compute/output any
desired solution diagnostic information with the signature

```C++
  int output_diagnostics(const realtype& t, const N_Vector w, const UserData& udata);
```

If no diagnostics information is desired, then this routine may just return 0.

Here, the `initial_conditions` routine will be called once when the simulation
begins, `external_forces` will be called on every evaluation of the ODE
right-hand side function for the Euler equations (it is assumed that this does
not require parallel communication or the results from `UserData::ExchangeStart`
/ `UserData::ExchangeEnd`), and `output_diagnostics` will be called at the same
frequency as the solution is output to disk.

To add a new executable using these auxiliary source code file(s), update
[src/CMakeLists.txt](./src/CMakeLists.txt) to include a new call to
`sundemo_add_executable` in a similar manner as the existing test problems e.g.,
`hurricane_yz.exe`.

## Authors

[Daniel R. Reynolds](http://faculty.smu.edu/reynolds)
