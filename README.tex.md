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

Steps showing the process to download this demo code, install the
relevant dependencies, and build the demo in a Linux or OS X
environment are as follows.  To compile this code you will need modern
C and C++ compilers.  All dependencies (SUNDIALS and KLU) for the demo
are installed in-place using [Spack](https://github.com/spack/spack).

```bash
> git clone https://github.com/drreynolds/sundials-manyvector-demo.git
> cd sundials-manyvector-demo
> git clone https://github.com/spack/spack.git .spack
> .spack/bin/spack install sundials +int64 +klu +mpi ~examples-f77 ~examples-install ~examples-c ~CVODE ~CVODES ~IDA ~IDAS ~KINSOL
> .spack/bin/spack view symlink libs sundials
> .spack/bin/spack view symlink mpi mpi
> make
```

The above steps will build all codes in 'production' mode, with
optimization enabled, and both OpenMP and debugging symbols turned
off. 

Alternately, if you already have MPI, SUNDIALS and KLU/SuiteSparse
installed, you can edit the file `Makefile.in` to specify these
installations, and skip the Spack-related steps above. 

Additionally, you may edit the `Makefile.opts` file to switch between
an optimized/debugging build, and to enable/disable OpenMP prior to
running `make` above. 



## (Current) Documentation

This test simulates a 3D nonlinear inviscid compressible Euler
equation, 
$$
  w_t = -\nabla\cdot F(w) + G
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
  w(t_0,X) = w_0(X)
$$
and face-specific boundary conditions, [xlbc, xrbc] x [ylbc, yrbc] x
[zlbc, zrbc], where each may be any one of 

* periodic (0),
* homogeneous Neumann (1), or
* homogeneous Dirichlet (2),

under the restriction that if any boundary is set to "periodic" then
the opposite face must also indicate a periodic condition.

Here, the 'solution' is given by
$w = \begin{bmatrix} \rho & \rho v_x & \rho v_y & \rho v_z & e_t\end{bmatrix}^T
= \begin{bmatrix} \rho & m_x & m_y & m_z & e_t\end{bmatrix}^T$,
that corresponds to the density, x,y,z-momentum, and the total energy
per unit volume.  The fluxes are given by 
$$
  F_x(w) = \begin{bmatrix} \rho v_x & \rho v_x^2 + p & \rho v_x v_y & \rho v_x v_z & v_x (e_t+p)\end{bmatrix}^T
$$
$$
  F_y(w) = \begin{bmatrix} \rho v_y & \rho v_x v_y & \rho v_y^2 + p & \rho v_y v_z & v_y (e_t+p)\end{bmatrix}^T
$$
$$
  F_z(w) = \begin{bmatrix} \rho v_z & \rho v_x v_z & \rho v_y v_z & \rho v_z^2 + p & v_z (e_t+p)\end{bmatrix}^T.
$$
The external force $G(X,t)$ is test-problem-dependent, and the ideal
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

We have the parameters:

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

Note: the above follows the description in section 7.3.1-7.3.3 of
https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html

This program solves the problem using a finite volume spatial
semi-discretization over a uniform grid of dimensions
`nx` x `ny` x `nz`, with fluxes calculated using a 5th-order WENO
reconstruction. The spatial domain uses a 3D domain decomposition
approach for parallelism over nprocs MPI processes, with layout
`npx` x `npy` x `npz` defined automatically via the `MPI_Dims_create` 
utility routine.  Each field is stored in its own parallel `N_Vector`
object; these are combined together to form the full "solution" vector
$w$ using the `MPIManyVector` `N_Vector` module.  The resulting
initial-value problem is solved using a temporally-adaptive explicit
Runge Kutta method from ARKode's ARKStep module.  The solution is
output to disk and solution statistics are optionally output to the
screen at specified frequencies, and run statistics are printed at the
end. 

Individual test problems may be uniquely specified through an input
file and an auxiliarly source code file that should be linked with
this main routine at compile time. 

The default input file name is `input_euler3D.txt`, however alternate
input file names may be specified on the command line (the first
argument following the executable name).  This input file contains:

* spatial domain, $\Omega$ -- `xl`, `xr`, `yl`, `yr`, `zl`, `zr`

* time interval, $(t_0,t_f]$ -- `t0`, `tf`

* the ratio of specific heats, $\gamma$ -- `gamma`

* spatial discretization dimensions -- `nx`, `ny`, `nz`

* boundary condition types -- `xlbc`, `xrbc`, `ylbc`, `yrbc`, `zlbc`, `zrbc`
  
* desired cfl fraction -- `cfl` (if set to zero, then the time step is
  chosen purely using temporal adaptivity).

* number of desired solution outputs -- `nout`

Additionally, this file contains the parameter `showstats`, a nonzero
value enables optional output of RMS averages for each field at the
same frequency as the solution is output to disk. 

The auxiliary source code file must contain three functions.  Each of
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

A second input file, `solve_params.txt` specifies all ARKode time
integration-related parameters that may be used to control the
simulation without recompilation.  Both the problem-specific input
file (nominally `input_euler3D.txt`) and `solve_params.txt` are
internally documented; however, for further information on the ARKode
solver parameters and the meaning of individual values, see the ARKode
documentation,
http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/index.html.



## Authors
[Daniel R. Reynolds](http://faculty.smu.edu/reynolds)
