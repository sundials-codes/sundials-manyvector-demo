# SUNDIALS ManyVector+Multirate Demo

[Note: this project is in active development; do not expect
executables to run correctly (or even compile) at present.]

This is a SUNDIALS-based demonstration application to assess and
demonstrate the large-scale parallel performance of new capabilities
that have been added to SUNDIALS in recent years.  Namely:

1. SUNDIALS' new MPIManyVector module, that allows extreme flexibility
   in how a solution "vector" is staged on computational resources.

2. ARKode's new multirate integration module, MRIStep, allowing
   high-order accurate calculations that subcycle "fast" processes
   within "slow" ones.

3. (eventually) SUNDIALS' new flexible linear solver interfaces, to
   enable streamlined use of scalable linear solver libraries (e.g.,
   *hypre*, PETSc and Trilinos).

Steps showing the process to download this demo code, install the
relevant dependencies, and build the demo in a Linux or OS X
environment are as follows.  To compile this code you will need modern
C and C++ compilers.  All dependencies (SUNDIALS, KLU and HDF5) for the demo
are installed in-place using [Spack](https://github.com/spack/spack).

```bash
   git clone https://github.com/drreynolds/sundials-manyvector-demo.git
   cd sundials-manyvector-demo
   git clone https://github.com/spack/spack.git .spack
   .spack/bin/spack install hdf5 +mpi +pic +szip
   .spack/bin/spack install sundials +int64 +klu +mpi ~examples-f77 ~examples-install ~examples-c ~CVODE ~CVODES ~IDA ~IDAS ~KINSOL
   .spack/bin/spack view symlink hdf5 hdf5
   .spack/bin/spack view symlink sundials sundials
   .spack/bin/spack view symlink mpi mpi
   make
```

The above steps will build all codes in 'production' mode, with
optimization enabled, and both OpenMP and debugging symbols turned
off.

Alternately, if you already have MPI, SUNDIALS, parallel HDF5, and
KLU/SuiteSparse installed, you can edit the file `Makefile.in` to
specify these installations, and skip the Spack-related steps above.

Additionally, you may edit the `Makefile.opts` file to switch between
an optimized/debugging build, and to enable/disable OpenMP prior to
running `make` above.  Also, HDF5-based I/O may be disabled entirely
(e.g., if HDF5 is unavailable on a given machine) by setting `USEHDF5
= 0` in `Makefile.opts`.



## (Current) Documentation

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
the `Makefile` when building `compile_test.exe`, and may be emulated
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
`Makefile` in a similar manner as the existing test problems are built
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
