# SUNDIALS Multirate+ManyVector Demo

[Note: this project is in active development; do not expect anything to run (or even compile) at present.]

This is a SUNDIALS-based demonstration application to assess and demonstrate the large-scale parallel performance of new capabilities that have been added to SUNDIALS in recent years.  Namely:

1. ARKode's new multirate integration module, MRIStep, allowing high-order accurate calculations that subcycle "fast" processes within "slow" ones.

2. SUNDIALS' new MPIManyVector module, that allows extreme flexibility in how a solution "vector" is staged on computational resources.

To run this demo you will need modern C and C++ compilers.  All dependencies (SUNDIALS and SuiteSparse) for the demo are installed in-place using Spack, which is included in this repository.

Steps showing the process to download this demo code, install the relevant dependencies, and build the demo in a Linux or OS X environment are as follows:

```bash
  $ git clone https://github.com/drreynolds/sundials-manyvector-demo.git
  $ cd sundials-manyvector-demo
  $ .spack/bin/spack install sundials +suite-sparse +mpi
  $ .spack/bin/spack view symlink libs sundials
  $ .spack/bin/spack view symlink mpi mpi
  $ make
```

## (Current) Documentation

This test simulates a 3D nonlinear inviscid compressible Euler equation,

  ![w_t = - Div(F(w)) + G†](https://latex.codecogs.com/svg.latex?w_t%20%3D%20-%20%5Cnabla%5Ccdot%20F%28w%29%20&plus;%20G%2C)

for t \in (t0, tf], X = (x,y,z) in \Omega = [`xl`,`xr`] x [`yl`,`yr`] x [`zl`,`zr`], with initial condition w(t0,X) = w0(X), and boundary conditions [`xlbc`,`xrbc`] x [`ylbc`,`yrbc`] x [`zlbc`,`zrbc`], where each may be any one of 

* periodic (0),
* homogeneous Neumann (1), or
* homogeneous Dirichlet (2),

under the restriction that if any boundary is set to "periodic" then the opposite face must also indicate a periodic condition.
    
Here, the 'solution' is given by

   ![w = [rho, rho*vx, rho*vy, rho*vz, E]^T = [rho, mx, my, mz, E]^T](https://latex.codecogs.com/svg.latex?w%20%3D%20%5B%5Crho%2C%20%5Crho%20v_x%2C%20%5Crho%20v_y%2C%20%5Crho%20v_z%2C%20e%5D%5ET%20%3D%20%5B%5Crho%2C%20m_x%2C%20m_y%2C%20m_z%2C%20e%5D%5ET)
   
that corresponds to the density, x,y,z-momentum, and the total energy per unit volume.  The fluxes are given by

  ![Fx(w) = [rho*vx, rho*vx^2+p, rho*vx*vy, rho*vx*vz, vx*(E+p)]^T](https://latex.codecogs.com/svg.latex?F_x%28w%29%20%3D%20%5B%5Crho%20v_x%2C%20%5Crho%20v_x%5E2&plus;p%2C%20%5Crho%20v_x%20v_y%2C%20%5Crho%20v_x%20v_z%2C%20v_x%20%28E&plus;p%29%5D%5ET)
  ![Fy(w) = [rho*vy, rho*vx*vy, rho*vy^2+p, rho*vy*vz, vy*(E+p)]^T](https://latex.codecogs.com/svg.latex?F_y%28w%29%20%3D%20%5B%5Crho%20v_y%2C%20%5Crho%20v_x%20v_y%2C%20%5Crho%20v_y%5E2&plus;p%2C%20%5Crho%20v_y%20v_z%2C%20v_y%20%28E&plus;p%29%5D%5ET)
  ![Fz(w) = [rho*vz, rho*vx*vz, rho*vy*vz, rho*vz^2+p, vz*(E+p)]^T](https://latex.codecogs.com/svg.latex?F_z%28w%29%20%3D%20%5B%5Crho%20v_z%2C%20%5Crho%20v_x%20v_z%2C%20%5Crho%20v_y%20v_z%2C%20%5Crho%20v_z%5E2&plus;p%2C%20v_z%20%28E&plus;p%29%5D%5ET)

the external force is test-problem-dependent, and the ideal gas equation of state gives

  ![p = R/cv*(E - rho/2*(vx^2+vy^2+vz^2)](https://latex.codecogs.com/svg.latex?p%20%3D%20%5Cfrac%7BR%7D%7Bc_v%7D%20%5Cleft%28E%20-%20%5Cfrac%7B%5Crho%7D%7B2%7D%28v_x%5E2&plus;v_y%5E2&plus;v_z%5E2%29%5Cright%29)

and

  ![E = p*cv/R + rho/2*(vx^2+vy^2+vz^2)](https://latex.codecogs.com/svg.latex?E%20%3D%20%5Cfrac%7Bp%20c_v%7D%7BR%7D%20&plus;%20%5Cfrac%7B%5Crho%7D%7B2%7D%28v_x%5E2&plus;v_y%5E2&plus;v_z%5E2%29)
  
or equivalently,

  ![p = (gamma-1)*(E - rho/2*(vx^2+vy^2+vz^2)](https://latex.codecogs.com/svg.latex?p%20%3D%20%28%5Cgamma-1%29%20%5Cleft%28E%20-%20%5Cfrac%7B%5Crho%7D%7B2%7D%28v_x%5E2&plus;v_y%5E2&plus;v_z%5E2%29%5Cright%29)
  
and

  ![E = p/(gamma-1) + rho/2*(vx^2+vy^2+vz^2)](https://latex.codecogs.com/svg.latex?E%20%3D%20%5Cfrac%7Bp%7D%7B%5Cgamma-1%7D%20&plus;%20%5Cfrac%7B%5Crho%7D%7B2%7D%28v_x%5E2&plus;v_y%5E2&plus;v_z%5E2%29)
  
We have the parameters

* R is the specific ideal gas constant (287.14 J/kg/K).
* ![cv](https://latex.codecogs.com/svg.latex?c_v) is the specific heat capacity at constant volume (717.5 J/kg/K),
* ![gamma](https://latex.codecogs.com/svg.latex?%5Cgamma) is the ratio of specific heats, ![gamma = cp/cv = 1 + R/cv](https://latex.codecogs.com/svg.latex?%5Cgamma%20%3D%20%5Cfrac%7Bc_p%7D%7Bc_v%7D%20%3D%201%20&plus;%20%5Cfrac%7BR%7D%7Bc_v%7D) (1.4),

corresponding to air (predominantly an ideal diatomic gas). The speed of sound in the gas is then given by

  ![c = sqrt(gamma*p/rho)](https://latex.codecogs.com/svg.latex?c%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Cgamma%20p%7D%7B%5Crho%7D%7D)

The fluid variables above are non-dimensionalized; in standard SI units these would be:

  [rho] = kg / m^3
  
  [vx] = [vy] = [vz] = m/s  =>  [mx] = [my] = [mz] = kg / m^2 / s
  
  [E] = kg / m / s^2

Note: the above follows the description in section 7.3.1-7.3.3 of https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html

This program solves the problem using a finite volume spatial semi-discretization over a uniform grid of dimensions nx x ny x nz, with fluxes calculated using a 5th-order WENO reconstruction.  The spatial domain uses a 3D domain decomposition approach for parallelism over nprocs MPI processes, with layout npx x npy x npz defined automatically via the MPI_Dims_create utility routine.  Each field is stored in its own parallel N_Vector object; these are combined together to form the full "solution" vector w using the MPIManyVector N_Vector module.  The resulting IVP is solved using a temporally-adaptive ERK method from ARKode's ARKStep module.  The solution is output to disk and solution statistics are optionally output to the screen at specified frequencies, and run statistics are printed at the end.

Individual test problems may be uniquely specified through an input file and an auxiliarly source code file that should be linked with this main routine at compile time.  

The input file, `input_euler3D.txt`, contains:

* the ratio of specific heats (`gamma`) 
* spatial domain Omega, (`xl`,`xr`,`yl`,`yr`,`zl`,`zr`)
* time interval (`t0`,`tf`)
* spatial discretization dimensions (`nx`,`ny`,`nz`) 
* boundary conditions (`xlbc`,`xrbc`,`ylbc`,`yrbc`,`zlbc`,`zrbc`)
* number of desired solution outputs (`nout`)

Additionally, this file contains the parameter `showstats`, a nonzero value enables optional output of RMS averages for each field at the same frequency as the solution is output to disk.

The auxiliary source code file must contain two functions:

* the initial condition function w0(X) must have the signature:

```C++
     int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);
```

* the forcing function G(X) must have the signature

```C++
     int external_forces(const realtype& t, N_Vector G, const UserData& udata);
```


## Authors
[Daniel R. Reynolds](mailto:reynolds@smu.edu)

