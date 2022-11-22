# SUNDIALS ManyVector+Multirate Demonstration Code

[Note: this project is in active development.]

This is a [SUNDIALS](https://github.com/LLNL/sundials)-based demonstration
application to assess and demonstrate the large-scale parallel performance of
new capabilities that have been added to SUNDIALS in recent years. Namely:

1. The new SUNDIALS [MPIManyVector](https://sundials.readthedocs.io/en/latest/nvectors/NVector_links.html#the-nvector-mpimanyvector-module)
   implementation, that enables flexibility in how a solution data is
   partitioned across computational resources e.g., CPUs and GPUs.

2. The new [ARKODE](https://sundials.readthedocs.io/en/latest/arkode/index.html)
   multirate integration module, MRIStep, allowing high-order accurate
   calculations that subcycle "fast" processes within "slow" ones.

3. The new flexible SUNDIALS [SUNLinearSolver](https://sundials.readthedocs.io/en/latest/sunlinsol/index.html)
   interfaces, to enable streamlined use of problem specific and scalable
   linear solver libraries e.g., SuiteSparse and MAGMA.

## Model Equations

This code simulates a 3D nonlinear inviscid compressible Euler equation with
advection and reaction of chemical species,

$$w_t = -\nabla\cdot F(w) + G(X,t,w),$$

for independent variables $(X,t) = (x,y,z,t) \in \Omega \times [t_0, t_f]$
where the spatial domain is a three-dimensional cube,
$\Omega = [x_l, x_r] \times [y_l, y_r] \times [z_l, z_r]$.

The differential equation is completed using initial condition
$w(X,t_0) = w_0(X)$ and face-specific boundary conditions may be periodic (0),
homogeneous Neumann (1), homogeneous Dirichlet (2), or reflecting (3) under the
restriction that if any boundary is set to "periodic" then the opposite face
must also indicate a periodic condition.

The system state vector $w$ is

$$w = \begin{bmatrix} \rho & \rho v_x & \rho v_y & \rho v_z & e_t & \mathbf{c} \end{bmatrix}^T = \begin{bmatrix} \rho & m_x & m_y & m_z & e_t & \mathbf{c} \end{bmatrix}^T$$

corresponding to the density, momentum in the x, y, and z directions, total
energy per unit volume, and any number of chemical densities
$\mathbf{c}\in\mathbb{R}^{nchem}$ that are advected along with the fluid. The
fluxes are given by

$$F_x(w) = \begin{bmatrix} \rho v_x & \rho v_x^2 + p & \rho v_x v_y & \rho v_x v_z & v_x (e_t+p) & \mathbf{c} v_x \end{bmatrix}^T,$$

$$F_y(w) = \begin{bmatrix} \rho v_y & \rho v_x v_y & \rho v_y^2 + p & \rho v_y v_z & v_y (e_t+p) & \mathbf{c} v_y \end{bmatrix}^T,$$

$$F_z(w) = \begin{bmatrix} \rho v_z & \rho v_x v_z & \rho v_y v_z & \rho v_z^2 + p & v_z (e_t+p) & \mathbf{c} v_z \end{bmatrix}^T.$$

The external force $G(X,t,w)$ is test-problem-dependent, and the ideal gas
equation of state gives $p = \frac{R}{c_v}(e_t - \frac{\rho}{2}(v_x^2 + v_y^2 + v_z^2))$
and $e_t = \frac{pc_v}{R} + \frac{\rho}{2}(v_x^2 + v_y^2 + v_z^2)$
or equivalently, $p = (\gamma-1) (e_t - \frac{\rho}{2} (v_x^2 + v_y^2 + v_z^2))$
and $e_t = \frac{p}{\gamma - 1}\frac{\rho}{2}(v_x^2 + v_y^2 + v_z^2)$.

We have the physical parameters:

* $R$ is the specific ideal gas constant (287.14 J/kg/K),

* $c_v$ is the specific heat capacity at constant volume (717.5 J/kg/K),

* $\gamma = c_p/c_v = 1 + R/c_v$ is the ratio of specific heats (1.4),

corresponding to air (predominantly an ideal diatomic gas). The speed
of sound in the gas is then given by $c = \sqrt{\dfrac{\gamma p}{\rho}}$.

The fluid variables above are non-dimensionalized; in standard SI units
these would be:

* $[\rho] = kg / m^3$,

* $[v_x] = [v_y] = [v_z] = m/s$, which implies $[m_x] = [m_y] = [m_z] = kg / m^2 / s$

* $[e_t] = kg / m / s^2$, and

* $[\mathbf{c}_i] = kg / m^3$

Note: the fluid portion of the description above follows the presentation
[here](https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html)
in sections 7.3.1 - 7.3.3.

## Discretization

We discretize this problem using the method of lines, where we first semi-discretize
in space using a regular finite volume grid with dimensions `nx` x `ny` x `nz`, with
fluxes at cell faces calculated using a 5th-order FD-WENO reconstruction.  MPI
parallelization is achieved using a standard 3D domain decomposition, using `nprocs`
MPI ranks, with layout `npx` x `npy` x `npz` defined automatically via the
`MPI_Dims_create` utility routine.  The minimum size for any dimension is 3, so
to run a two-dimensional test in the yz-plane, one could specify `nx = 3` and
`ny = nz = 200`.  When run in parallel, only "active" spatial dimensions (those
with extent greater than 3) will be parallelized.

The fluid fields $\rho$, $m_x$, $m_y$, $m_z$, and $e_t$ are stored in separate serial
`N_Vector` objects on each MPI rank. The chemical species at all spatial locations over
each MPI rank are collocated into a single serial or RAJA `N_Vector` object when
running on the CPU or GPU respectively. The five fluid vectors and the chemical
species vector are combined together to form the full "solution" vector $w$ using
the `MPIManyVector` `N_Vector` module.

After spatial semi-discretization, we are faced with a large IVP system,

$$w'(t) = f_1(w) + f_2(w), \quad w(t_0)=w_0,$$

where $f_1(w)$ and $f_2(w)$ contain the spatially discretized forms of
$-\nabla\cdot F(w)$ and $G(X,t,w)$, respectively.

For non-reactive flows, the resulting initial-value problem is evolved in time
using an adaptive step explicit Runge-Kutta method from the ARKStep module in
ARKODE. For problems involving (typically stiff) chemical reactions, the problem
may be solved using one of two approaches.

1. It may be treated as a multirate initial-value problem, that is solved using
   the MRIStep module in ARKODE, wherein the gas dynamics equations are evolved
   explicitly at the slow time scale, while the chemical kinetics are evolved
   at a faster time scale using a temporally-adaptive, diagonally-implicit
   Runge-Kutta method from the ARKStep module.

2. It may be treated using mixed implicit-explicit (IMEX) methods at a single
   time scale.  Here, the gas dynamics equations are treated explicitly, while
   the chemical kinetics are treated implicitly, using an additive Runge-Kutta
   method from the ARKStep module.

For (1) we use SUNDIALS' modified Newton solver to handle the global nonlinear
algebraic systems arising at each implicit stage of each time step.  Since only
$f_2$ is treated implicitly and the reactions are purely local in space, the
Newton linear systems are block-diagonal. As such, we provide a custom
`SUNLinearSolver` implementation that solves each MPI rank-local linear system
independently. The portion of the Jacobian matrix on each rank is itself
block-diagonal. We further leverage this structure by solving each rank-local
linear system using either the sparse KLU (CPU-only) or batched dense MAGMA
(GPU-enabled) SUNDIALS `SUNLinearSolver` implementations.

The multirate approach (2) can leverage the structure of $f_2$ at a higher
level. Since the MRI method applied to this problem evolves "fast" sub-problems
of the form

$$v'(t) = f_2(t,v) + r_i(t), \quad i=2,\ldots,s,$$

and all MPI communication necessary to construct the forcing functions, $r_i(t)$,
has already been performed, each sub-problem consists of `nx` x `ny` x `nz`
spatially-decoupled fast IVPs. We construct a custom fast integrator that groups
all the independent fast IVPs on an MPI rank together as a single system evolved
using a rank-local ARKStep instance.  The code for this custom integrator itself
is minimal, primarily consisting of steps to access the local subvectors in $w$
on a given MPI rank and wrapping them in MPI-unaware ManyVectors provided to the
local ARKStep instance. The collection of independent local IVPs also leads to a
block diagonal Jacobian, and we again utilize the `SUNLinearSolver` modules listed
above for linear systems that arise within the modified Newton iteration.

## Installation

The following steps describe how to build the demonstation code in a Linux or OS X
environment.


### Gettting the Code

To obtain the code, clone this repository with Git:

```bash
  git clone https://github.com/sundials-codes/sundials-manyvector-demo.git
```

### Requirements

To compile the code you will need:

* [CMake](https://cmake.org) 3.18 or newer

* modern C and C++ compilers

* the NVIDIA [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (when
  using the CUDA backend)

* an MPI library e.g., [OpenMPI](https://www.open-mpi.org/),
  [MPICH](https://www.mpich.org/), etc.

* the [HDF5](https://www.hdfgroup.org/) high-performance data management and
  storage suite

* the [RAJA](https://github.com/LLNL/RAJA) performance portability library

* the [SUNDIALS](https://computing.llnl.gov/projects/sundials) library of time
  integrators and nonlinear solvers

* the [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) library
  of sparse direct linear solvers (when using a CPU backend)

* the [MAGMA](https://icl.utk.edu/magma/) dense linear solver "multicore+GPU"
  library (when using a GPU backend)

### Installing Dependencies

Many of the above dependencies can be installed using the
[Spack](https://spack.io/) package manager. For information on using Spack see
the getting started [guide](https://spack.readthedocs.io/en/latest/getting_started.html#getting-started).

Once Spack is setup, we recommend creating a Spack [environment](https://spack.readthedocs.io/en/latest/environments.html#)
with the required dependencies e.g., on a system with Pascal GPUs:

```bash
spack env create sundials-demo
spack env activate sundials-demo
spack add sundials +openmp +klu +magma +raja +cuda cuda_arch=60 ^magma +cuda cuda_arch=60 ^raja +cuda cuda_arch=60
spack add hdf5 +hl
spack install
```

To assist in building the dependencies on select systems the [spack](./spack)
directory contains environment files leveraging software already available on
the system. For example, on the OLCF Summit system:

```bash
module load gcc/10.2.0
module load cuda/11.4.2
module load cmake/3.21.3
cd spack
spack env create sundials-demo spack-summit.yaml
spack env activate sundials-demo
spack install
```

### Configuration Options

Once the necessary dependencies are installed, the following CMake variables can
be used to configure the demonstration code build:

* `CMAKE_INSTALL_PREFIX` - the path where executables and input files should be
  installed e.g., `my/install/path`. The executables will be installed in the
  `bin` directory and input files in the `tests` directory under the given path.

* `CMAKE_C_COMPILER` - the C compiler to use e.g., `mpicc`. If not set, CMake
  will attempt to automatically detect the C compiler.

* `CMAKE_C_FLAGS` - the C compiler flags to use e.g., `-g -O2`.

* `CMAKE_C_STANDARD` - the C standard to use, defaults to `99`.

* `CMAKE_CXX_COMPILER` - the C++ compiler to use e.g., `mpicxx`. If not set,
  CMake will attempt to automatically detect the C++ compiler.

* `CMAKE_CXX_FLAGS` - the C++ flags to use e.g., `-g -O2`.

* `CMAKE_CXX_STANDARD` - the C++ standard to use, defaults to `11`.

* `RAJA_ROOT` - the root directory of the RAJA installation, defaults to the
  value of the `RAJA_ROOT` environment variable. If not set, CMake will attempt
  to automaticall locate a RAJA install on the system.

* `RAJA_BACKEND` - the RAJA backend to use with the demonstration code, defaults
   to `SERIAL`. Supported options are `SERIAL` and `CUDA`.

* `SUNDIALS_ROOT` - the root directory of the SUNDIALS installation, defaults to
  the value of the `SUNDIALS_ROOT` environment variable. If not set, CMake will
  attempt to automatically locate a SUNDIALS install on the system.

* `ENABLE_HDF5` - build with HDF5 I/O support, defaults to `OFF`.

* `HDF5_ROOT` - the root directory of the HDF5 installation, defaults to the
  value of the `HDF5_ROOT` environment variable. If not set, CMake will attempt
  to automatically locate a HDF5 install on the system.

When RAJA is installed with CUDA support enabled, the following additional
variables may also be set:

* `CMAKE_CUDA_COMPILER` - the CUDA compiler to use e.g., `nvcc`. If not set,
  CMake will attempt to automatically detect the CUDA compiler.

* `CMAKE_CUDA_FLAGS` - the CUDA compiler flags to use.

* `CMAKE_CUDA_ARCHITECTURES` - the CUDA architecture to target e.g., `70`.

### Building

In-source builds are not permitted, as such the code should be configured and
built from a separate build directory e.g.,

```bash
  cd sundials-manyvector-demo
  mkdir build
  cd build
  cmake ../. \
    -DCMAKE_INSTALL_PREFIX="my/install/path/sundials-demo" \
    -DRAJA_BACKEND="SERIAL" \
    -DENABLE_HDF5="ON"
  make
  make install
```

## Running

Several test cases are included with the code and the necessary input files for
each case are contained in the subdirectories within the [tests](./tests)
directory. Each input file is internally documented to discuss all possible
input parameters (in case some have been added since this `README` was last
updated).

The input files contain parameters to set up the physical problem:

* spatial domain, $\Omega$ -- `xl`, `xr`, `yl`, `yr`, `zl`, `zr`

* time interval, $[t_0, t_f]$ -- `t0`, `tf`

* the ratio of specific heats, $\gamma$ -- `gamma`

* spatial discretization dimensions -- `nx`, `ny`, `nz`

* boundary condition types -- `xlbc`, `xrbc`, `ylbc`, `yrbc`, `zlbc`, `zrbc`

Parameters to control the execution of the code:

* desired cfl fraction -- `cfl` (if set to zero, then the time step is chosen purely using temporal adaptivity).

* number of desired solution outputs -- `nout`

* a flag to enable optional output of RMS averages for each field at the frequency specified via `nout` -- `showstats`

Numerous parameters are also provided to control how time integration is
performed (these are passed directly to ARKODE). For further information on the
ARKODE solver parameters and the meaning of individual values, see the
[ARKODE documentation](https://sundials.readthedocs.io/en/latest/index.html).

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

The parallel HDF5 solution snapshots are written at the frequency specified by
`nout`.  Accompanying these `output-#######.hdf5` files is an automatically
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
failure (nonzero). The initial condition function $w_0(X)$ must have the signature

```C++
  int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);
```

and the forcing function $G(X,t,w)$ must have the signature

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
not require the results from (`UserData::ExchangeStart`
/ `UserData::ExchangeEnd`), and `output_diagnostics` will be called at the same
frequency as the solution is output to disk.

To add a new executable using these auxiliary source code file(s), update
[src/CMakeLists.txt](./src/CMakeLists.txt) to include a new call to
`sundemo_add_executable` in a similar manner as the existing test problems e.g.,
`hurricane_yz.exe`.

## Authors

[Daniel R. Reynolds](https://people.smu.edu/dreynolds) and
[David J. Gardner](https://people.llnl.gov/gardner48)
