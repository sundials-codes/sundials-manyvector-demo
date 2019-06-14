# SUNDIALS Multirate+ManyVector Demo

[Note: this project is in active development; do not expect anything to run (or even compile) at present.]

This is a SUNDIALS-based demonstration application to assess and demonstrate the large-scale parallel performance of new capabilities that have been added to SUNDIALS in recent years.  Namely:

1. ARKode's new multirate integration module, MRIStep, allowing high-order accurate calculations that subcycle "fast" processes within "slow" ones.

2. SUNDIALS' new MPIManyVector module, that allows extreme flexibility in how a solution "vector" is staged on computational resources.

To run this demo you will need modern C and C++ compilers.  All dependencies (SUNDIALS and SuiteSparse) for the demo are installed in-place using Spack, which is included in this repository.

Steps showing the process to download this demo code, install the relevant dependencies, and build the demo in a Linux or OS X environment are as follows:

  $ git clone https://github.com/drreynolds/sundials-manyvector-demo.git
  $ cd sundials-manyvector-demo
  $ .spack/bin/spack install sundials +suite-sparse +mpi
  $ .spack/bin/spack view symlink libs sundials
  $ .spack/bin/spack view symlink mpi mpi
  $ make

## (Current) Documentation

This test simulates a 3D nonlinear inviscid compressible Euler equation,

  w_t = - Div(F(w)) + G,
   
for t in [t0, tf], X = (x,y,z) in Omega = [xl,xr] x [yl,yr] x [zl,zr], with initial condition

  w(t0,X) = w0(X),

and boundary conditions [xlbc,xrbc] x [ylbc,yrbc] x [zlbc,zrbc], where each may be any one of 

* periodic (0),
* homogeneous Neumann (1), or
* homogeneous Dirichlet (2),

under the restriction that if any boundary is set to "periodic" then the opposite face must also indicate a periodic condition.
    
Here, the state w = [rho, rho*vx, rho*vy, rho*vz, E]^T = [rho, mx, my, mz, E]^T, corresponding to the density, x,y,z-momentum, and the total energy per unit volume.  The fluxes are given by

  Fx(w) = [rho*vx, rho*vx^2+p, rho*vx*vy, rho*vx*vz, vx*(E+p)]^T
  Fy(w) = [rho*vy, rho*vx*vy, rho*vy^2+p, rho*vy*vz, vy*(E+p)]^T
  Fz(w) = [rho*vz, rho*vx*vz, rho*vy*vz, rho*vz^2+p, vz*(E+p)]^T

the external force is given by

  G(X) = [0, gx(X), gy(X), gz(X), 0]^T

and the ideal gas equation of state gives

  p = R/cv*(E - rho/2*(vx^2+vy^2+vz^2), and
  E = p*cv/R + rho/2*(vx^2+vy^2+vz^2),
  
or equivalently,

  p = (gamma-1)*(E - rho/2*(vx^2+vy^2+vz^2), and
  E = p/(gamma-1) + rho/2*(vx^2+vy^2+vz^2),
  
We have the parameters

* R is the specific ideal gas constant (287.14 J/kg/K).
* cv is the specific heat capacity at constant volume (717.5 J/kg/K),
* gamma is the ratio of specific heats, cp/cv = 1 + R/cv (1.4),

corresponding to air (predominantly an ideal diatomic gas). The speed of sound in the gas is then given by

  c = sqrt(gamma*p/rho).

The fluid variables above are non-dimensionalized; in standard SI units these would be:

  [rho] = kg / m^3
  [vx] = [vy] = [vz] = m/s  =>  [rho*vx] = kg / m^2 / s
  [E] = kg / m / s^2

Note: the above follows the description in section 7.3.1-7.3.3 of https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html

This program solves the problem using a finite volume spatial semi-discretization over a uniform grid of dimensions nx x ny x nz, with fluxes calculated using a 5th-order WENO reconstruction.  The spatial domain uses a 3D domain decomposition approach for parallelism over nprocs MPI processes, with layout npx x npy x npz defined automatically via the MPI_Dims_create utility routine.  Each field is stored in its own parallel N_Vector object; these are combined together to form the full "solution" vector w using the MPIManyVector N_Vector module.  The resulting IVP is solved using a temporally-adaptive ERK method from ARKode's ARKStep module.  The solution is output to disk and solution statistics are optionally output to the screen at specified frequencies, and run statistics are printed at the end.

Individual test problems may be uniquely specified through an input file and an auxiliarly source code file that should be linked with this main routine at compile time.  

The input file, input_euler3D.txt, contains:

* the ratio of specific heats gamma, 
* spatial domain Omega, 
* time interval (t0,tf], 
* spatial discretization dimensions (nx,ny,nz), 
* boundary conditions (xlbc,xrbc,ylbc,yrbc,zlbc,zrbc),
* nout indicates the number of desired solution outputs.

Additionally, this file contains the parameter 'showstats', a nonzero value enables optional output of RMS averages for each field at the same frequency as the solution is output to disk.

The auxiliary source code file must contain two functions:

* the initial condition function w0(X) must have the signature:

  int initial_conditions(realtype t, N_Vector w, UserData& udata)

* the forcing function G(X) must have the signature

  int external_forces(realtype t, N_Vector G, UserData& udata)



## Authors
[Daniel R. Reynolds](mailto:reynolds@smu.edu)

