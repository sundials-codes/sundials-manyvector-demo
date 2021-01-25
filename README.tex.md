# SUNDIALS ManyVector+Multirate Demo

[Note: this project is in active development.]

This is a SUNDIALS-based demonstration application to assess and demonstrate the
large-scale parallel performance of new capabilities that have been added to
SUNDIALS in recent years. Namely:

1. The new SUNDIALS MPIManyVector module, that allows extreme flexibility in how
   a solution "vector" is staged on computational resources.

2. The new ARKODE multirate integration module, MRIStep, allowing high-order
   accurate calculations that subcycle "fast" processes within "slow" ones.

3. The new flexible SUNDIALS linear solver interfaces, to enable streamlined use
   of scalable linear solver libraries (e.g., *hypre*, PETSc and Trilinos).

## Building

Steps showing the process to download this demo code, install the relevant
dependencies, and build the demo in a Linux or OS X environment are as follows.

To compile this code you will need:

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

All of the dependencies for this demo code can be installed using the
[Spack](https://github.com/spack/spack) package management tool e.g.,

```bash
   git clone https://github.com/spack/spack.git
   spack/bin/spack install mpi
   spack/bin/spack install hdf5 +mpi +pic +szip
   spack/bin/spack isntall suitesparse
   spack/bin/spack install raja +cuda
   spack/bin/spack install sundials +klu +mpi +raja +cuda
```

Alternately, the `scripts` directory contains shell scripts for setting up the
environment on various systems and installing some of the required libraries
(SUNDIALS, SuiteSparse, and RAJA).

The following CMake variables can be used to configure the build, enable various
options, and specify the location of the external libraries:

* `CMAKE_INSTALL_PREFIX` - the path where executables and input files should be
  installed e.g., `path/to/myinstall/`. The executables will be installed in the
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

* `RAJA_BACKEND` - set the RAJA backend to use in the demo code, defaults to
   `CUDA`

* `ENABLE_HDF5` - build with HDF5 I/O support, defaults to `OFF`

* `HDF5_ROOT` - the root directory of the HDF5 installation, defaults to the
  value of the `HDF5_ROOT` environment variable

* `CMAKE_CUDA_ARCHITECTURES` - the CUDA architecture to target, defaults to `70`

For example the following the following commands can be used to download and
build the demo code with RAJA support targeting NVIDIA Tesla V100 GPUs:

```bash
   git clone https://github.com/sundials-codes/sundials-manyvector-demo.git
   cd sundials-manyvector-demo
   mkdir build
   cd build
   cmake ../. \
     -DCMAKE_INSTALL_PREFIX="path/to/myworkspace" \
     -DCMAKE_C_COMPILER=mpicc \
     -DCMAKE_C_FLAGS="-g -O2" \
     -DCMAKE_CXX_COMPILER=mpicxx \
     -DCMAKE_CXX_FLAGS="-g -O2" \
     -DSUNDIALS_ROOT="path/to/mylibs/sundials-5.6.1" \
     -ENABLE_RAJA="ON" \
     -DRAJA_ROOT="path/to/mylibs/raja-0.13.0" \
     -DENABLE_HDF5="ON" \
     -DHDF5_ROOT="path/to/mylibs/hdf5-1.10.4"
  make
  make install
```

## Documentation

This code simulates a 3D nonlinear inviscid compressible Euler
equation with advection and reaction of chemical species,
$$
  w_t = -\nabla\cdot F(w) + G(X,t,w)
$$
for independent variables
$$
  (X,t) = (x,y,z,t) \in \Omega \times (t_0, t_f]
$$
where the spatial domain is a three-dimensional cube,
$$
  \Omega = [x_l, x_r] \times [y_l, y_r] \times [z_l,z_r]
$$
The differential equation is completed using initial condition
$$
  w(X,t_0) = w_0(X)
$$
and face-specific boundary conditions, [xlbc, xrbc] x [ylbc, yrbc] x
[zlbc, zrbc], where each may be any one of

* periodic (0),
* homogeneous Neumann (1),
* homogeneous Dirichlet (2), or
* reflecting (3)

under the restriction that if any boundary is set to "periodic" then
the opposite face must also indicate a periodic condition.

Here, the 'solution' is given by
$w = \begin{bmatrix} \rho & \rho v_x & \rho v_y & \rho v_z & e_t & \mathbf{c} \end{bmatrix}^T
= \begin{bmatrix} \rho & m_x & m_y & m_z & e_t & \mathbf{c}\end{bmatrix}^T$,
that corresponds to the density, x,y,z-momentum, total energy
per unit volume, and any number of chemical densities
$\mathbf{c}\in\mathbb{R}^{nchem}$ that are advected along with the
fluid.  The fluxes are given by
$$
  F_x(w) = \begin{bmatrix} \rho v_x & \rho v_x^2 + p & \rho v_x v_y & \rho v_x v_z & v_x (e_t+p) & \mathbf{c} v_x \end{bmatrix}^T
$$
$$
  F_y(w) = \begin{bmatrix} \rho v_y & \rho v_x v_y & \rho v_y^2 + p & \rho v_y v_z & v_y (e_t+p) & \mathbf{c} v_y \end{bmatrix}^T
$$
$$
  F_z(w) = \begin{bmatrix} \rho v_z & \rho v_x v_z & \rho v_y v_z & \rho v_z^2 + p & v_z (e_t+p) & \mathbf{c} v_z \end{bmatrix}^T.
$$
The external force $G(X,t,w)$ is test-problem-dependent, and the ideal
gas equation of state gives
$$
  p = \frac{R}{c_v}\left(e_t - \frac{\rho}{2} (v_x^2 + v_y^2 + v_z^2)\right)
$$
and
$$
  e_t = \frac{p c_v}{R} + \frac{\rho}{2}(v_x^2 + v_y^2 + v_z^2),
$$
or equivalently,
$$
  p = (\gamma-1) \left(e_t - \frac{\rho}{2} (v_x^2 + v_y^2 + v_z^2)\right)
$$
and
$$
  e_t = \frac{p}{\gamma-1} + \frac{\rho}{2}(v_x^2 + v_y^2 + v_z^2).
$$

We have the physical parameters:

* R is the specific ideal gas constant (287.14 J/kg/K).

* $c_v$ is the specific heat capacity at constant volume (717.5
  J/kg/K),

* $\gamma$ is the ratio of specific heats, $\gamma = \frac{c_p}{c_v} =
  1 + \frac{R}{c_v}$ (1.4),

corresponding to air (predominantly an ideal diatomic gas). The speed
of sound in the gas is then given by
$$
  c = \sqrt{\frac{\gamma p}{\rho}}
$$
The fluid variables above are non-dimensionalized; in standard SI
units these would be:

* [rho] = kg / m$^3$,

* [vx] = [vy] = [vz] = m/s, which implies that [mx] = [my] = [mz] = kg / m$^2$ / s, and

* [et] = kg / m / s$^2$

* [\mathbf{c}_i] = kg / m$^3$

Note: the fluid portion of the above description follows section 7.3.1-7.3.3 of
https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html

This program solves the above problem using a finite volume spatial
semi-discretization over a uniform grid of dimensions
`nx` x `ny` x `nz`, with fluxes calculated using a 5th-order FD-WENO
reconstruction. The spatial domain uses a 3D domain decomposition
approach for parallelism over `nprocs` MPI processes, with layout
`npx` x `npy` x `npz` defined automatically via the `MPI_Dims_create`
utility routine.  The minimum size for any dimension is 3, so to run a
two-dimensional test in the yz-plane, one could specify `nx=3` and
`ny=nz=200` -- when run in parallel, only 'active' spatial dimensions
(those with extent greater than 3) will be parallelized.  Each fluid
field ($\rho$, $m_x$, $m_y$, $m_z$ and $e_t$) is stored in its own
parallel `N_Vector` object.  Chemical species at all spatial
locations over a single MPI rank are collocated into a single serial
`N_Vector` object.  The five fluid vectors and the chemical species
vector are combined together to form the full "solution" vector $w$
using the `MPIManyVector` `N_Vector` module.  For non-reactive flows,
the resulting initial-value problem is solved using a
temporally-adaptive explicit Runge Kutta method from ARKode's ARKStep
module.  For problems involving [typically stiff] chemical reactions,
the multirate initial-value problem is solved using ARKode's MRIStep
module, wherein the gas dynamics equations are evolved explicitly at
the 'slow' time scale, while the chemical kinetics are evolved
using a temporally-adaptive, diagonally-implicit Runge--Kutta method
from ARKode's ARKStep module.  These MPI rank-local implicit systems
are solved using the default modified Newton nonlinear solver, with a
custom linear solver that solves each rank-local linear system using
the SUNLinSol_KLU sparse-direct linear solver module.  Solutions are
output to disk using parallel HDF5, solution statistics are
optionally output to the screen at specified frequencies, and run
statistics are printed at the end of the simulation.

Individual test problems are uniquely specified through an input
file and auxiliarly source code file(s) that should be linked with
this main routine at compile time.  By default, all codes are built
with no chemical species; however, this may be controlled at
compilation time using the `NVAR` preprocessor directive,
corresponding to the number of unknowns at any spatial location.
Hence, the [default] minimum value for `NVAR` is 5, so for a
calculation with 4 chemical species the code should be compiled with
the preprocessor flag `-DNVAR=9`.  An example of this is provided in
`src/CMakeLists.txt` when building `compile_test.exe`, and may be emulated
for user-defined problems.

Example input files are provided in the `inputs/` folder -- these are
internally documented to discuss all possible input parameters (in
case some have been added since this `README` was last updated).  To
specify an input file to the executable, the input filename should be
provided using the `-f` flag, e.g.
```bash
   <executable> -f <input_file>
```
This input file contains parameters to set up the physical problem:

* spatial domain, $\Omega$ -- `xl`, `xr`, `yl`, `yr`, `zl`, `zr`

* time interval, $(t_0,t_f]$ -- `t0`, `tf`

* the ratio of specific heats, $\gamma$ -- `gamma`

* spatial discretization dimensions -- `nx`, `ny`, `nz`

* boundary condition types -- `xlbc`, `xrbc`, `ylbc`, `yrbc`, `zlbc`, `zrbc`

parameters to control the execution of the code:

* desired cfl fraction -- `cfl` (if set to zero, then the time step is
  chosen purely using temporal adaptivity).

* number of desired solution outputs -- `nout`

* a flag to enable optional output of RMS averages for each field at
  the frequency spefied via `nout` -- `showstats`

as well as parameters to control how time integration is performed
(these are passed directly to ARKode).  For further information on the
ARKode solver parameters and the meaning of individual values, see the
ARKode documentation,
http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/index.html.

Additionally, any input parameters may also be specified on the
command line, e.g.
```bash
   <executable> --nx=100 --ny=100 --nz=400
```

The auxiliary source code files must contain three functions.  Each of
these must return an integer flag indicating success (0) or failure
(nonzero). The initial condition function $w_0(X)$ must have the
signature:

```C++
   int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);
```

and the forcing function $G(X,t)$ must have the signature

```C++
   int external_forces(const realtype& t, N_Vector G, const UserData& udata);
```

Additionally, a function must be supplied to compute/output any
desired solution diagnostic information:

```C++
   int output_diagnostics(const realtype& t, const N_Vector w, const UserData& udata);
```

If no diagnostics information is desired, then this routine may just
return 0.

Here, the `initial_conditions` routine will be called once when the
simulation begins, `external_forces` will be called on every
evaluation of the ODE right-hand side function for the Euler
equations (it is assumed that this does not require parallel
communication, or the results from `UserData::ExchangeStart` /
`UserData::ExchangeEnd`), and `output_diagnostics` will be called at
the same frequency as the solution is output to disk.

To supply these auxiliary source code file(s), add this to the
`src/CMakeLists.txt` in a similar manner as the existing test problems are built
(e.g. `hurricane_yz.exe`).

As stated above, this code uses parallel HDF5 to store solution
snapshots at the frequency specified by `nout`.  Accompanying these
`output-#######.hdf5` files is an automatically-generated input file,
`restart_parameters.txt` that stores a complete set of input
parameters to restart the simulation from the most recently-generated
output file.  This is a "warm" restart, in that it will pick up the
calculation where the previous one left off, using the same initial
time step size as ARKStep would use.  This restart may differ slightly
from an uninterrupted run since other internal ARKStep time adaptivity
parameters cannot be reused.  We note that the restart must use the
same spatial grid size and number of chemical tracers as the original
run, but it may use a different number of MPI tasks if desired.


## Authors
[Daniel R. Reynolds](http://faculty.smu.edu/reynolds)
