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
<p align="center"><img src="/tex/d9863f826c2348e733804a7429c419ad.svg?invert_in_darkmode&sanitize=true" align=middle width=148.6959507pt height=16.438356pt/></p>
for independent variables
<p align="center"><img src="/tex/b1fb667c457471aa9a6fe9b8a16c5c51.svg?invert_in_darkmode&sanitize=true" align=middle width=227.07922545pt height=17.031940199999998pt/></p>
where the spatial domain is a three-dimensional cube,
<p align="center"><img src="/tex/724344359d58ffb008954bfadd390bc2.svg?invert_in_darkmode&sanitize=true" align=middle width=210.46034459999998pt height=16.438356pt/></p>
The differential equation is completed using initial condition 
<p align="center"><img src="/tex/bbc622f3dae0f361808755f8d67af6a7.svg?invert_in_darkmode&sanitize=true" align=middle width=129.27608594999998pt height=16.438356pt/></p>
and face-specific boundary conditions, [xlbc, xrbc] x [ylbc, yrbc] x
[zlbc, zrbc], where each may be any one of 

* periodic (0),
* homogeneous Neumann (1), or
* homogeneous Dirichlet (2),

under the restriction that if any boundary is set to "periodic" then
the opposite face must also indicate a periodic condition.

Here, the 'solution' is given by
<img src="/tex/27160d5a22a8a7276b9c9ee8e285f26a.svg?invert_in_darkmode&sanitize=true" align=middle width=418.92522045pt height=35.5436301pt/>,
that corresponds to the density, x,y,z-momentum, and the total energy
per unit volume.  The fluxes are given by 
<p align="center"><img src="/tex/edde56579807923c3cd19bd5843fec36.svg?invert_in_darkmode&sanitize=true" align=middle width=384.32648264999995pt height=23.5253469pt/></p>
<p align="center"><img src="/tex/b06018efc55bb0d7f41c8280c3044cb7.svg?invert_in_darkmode&sanitize=true" align=middle width=382.5153948pt height=23.9085792pt/></p>
<p align="center"><img src="/tex/f83cbef467a49d5ff0921ac1612ce3ca.svg?invert_in_darkmode&sanitize=true" align=middle width=389.33422275pt height=23.5253469pt/></p>
The external force <img src="/tex/c441e18e502be64ac772003edac839dc.svg?invert_in_darkmode&sanitize=true" align=middle width=52.94748029999999pt height=24.65753399999998pt/> is test-problem-dependent, and the ideal
gas equation of state gives 
<p align="center"><img src="/tex/272a02648ae7ce4db259539aa98655dc.svg?invert_in_darkmode&sanitize=true" align=middle width=218.4856146pt height=36.09514755pt/></p>
and
<p align="center"><img src="/tex/fcab1314e35d6d578ef90227b587c829.svg?invert_in_darkmode&sanitize=true" align=middle width=200.67741375pt height=29.47417935pt/></p>
or equivalently,
<p align="center"><img src="/tex/c525acb8ade640e26804cc23100fc642.svg?invert_in_darkmode&sanitize=true" align=middle width=250.13614230000002pt height=30.1801401pt/></p>
and
<p align="center"><img src="/tex/bb93cc09da34e3117795a4e98a4e4db6.svg?invert_in_darkmode&sanitize=true" align=middle width=215.21714114999997pt height=32.6705313pt/></p>

We have the parameters:

* R is the specific ideal gas constant (287.14 J/kg/K).

* <img src="/tex/aa8cfea83e4502fbd685d6c095494147.svg?invert_in_darkmode&sanitize=true" align=middle width=14.102064899999991pt height=14.15524440000002pt/> is the specific heat capacity at constant volume (717.5
  J/kg/K),
  
* <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> is the ratio of specific heats, <img src="/tex/f8415659af3e4a9e110591f46cc2875e.svg?invert_in_darkmode&sanitize=true" align=middle width=113.34245999999997pt height=28.670654099999997pt/> (1.4), 

corresponding to air (predominantly an ideal diatomic gas). The speed
of sound in the gas is then given by 
<p align="center"><img src="/tex/e55dd025376e04a1ada428d642da3089.svg?invert_in_darkmode&sanitize=true" align=middle width=67.10942039999999pt height=39.452455349999994pt/></p>
The fluid variables above are non-dimensionalized; in standard SI
units these would be: 

* [rho] = kg / m<img src="/tex/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>,

* [vx] = [vy] = [vz] = m/s, which implies that [mx] = [my] = [mz] = kg / m<img src="/tex/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/> / s, and

* [et] = kg / m / s<img src="/tex/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>

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
<img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> using the `MPIManyVector` `N_Vector` module.  The resulting
initial-value problem is solved using a temporally-adaptive explicit
Runge Kutta method from ARKode's ARKStep module.  The solution is
output to disk and solution statistics are optionally output to the
screen at specified frequencies, and run statistics are printed at the
end. 

Individual test problems may be uniquely specified through an input
file and an auxiliarly source code file that should be linked with
this main routine at compile time. 

The input file, `input_euler3D.txt`, contains:

* the ratio of specific heats, <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> -- `gamma`

* spatial domain, <img src="/tex/9432d83304c1eb0dcb05f092d30a767f.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/> -- `xl`, `xr`, `yl`, `yr`, `zl`, `zr`

* time interval, <img src="/tex/bbde6652efaeb60e967ee67be6440eb7.svg?invert_in_darkmode&sanitize=true" align=middle width=46.033257599999985pt height=24.65753399999998pt/> -- `t0`, `tf`

* spatial discretization dimensions -- `nx`, `ny`, `nz`

* boundary condition types -- `xlbc`, `xrbc`, `ylbc`, `yrbc`, `zlbc`, `zrbc`
  
* number of desired solution outputs -- `nout`

Additionally, this file contains the parameter `showstats`, a nonzero
value enables optional output of RMS averages for each field at the
same frequency as the solution is output to disk. 

The auxiliary source code file must contain three functions.  Each of
these must return an integer flag indicating success (0) or failure
(nonzero). The initial condition function <img src="/tex/d3cb4393199b89ca003e78d3486fa147.svg?invert_in_darkmode&sanitize=true" align=middle width=46.837068299999984pt height=24.65753399999998pt/> must have the
signature: 

```C++
   int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);
```

and the forcing function <img src="/tex/c441e18e502be64ac772003edac839dc.svg?invert_in_darkmode&sanitize=true" align=middle width=52.94748029999999pt height=24.65753399999998pt/> must have the signature

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
simulation without recompilation.  Both `input_euler3D.txt` and
`solve_params.txt` are internally documented; however, for further
information on the ARKode solver parameters and the meaning of
individual values, see the ARKode documentation,
http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/index.html.



## Authors
[Daniel R. Reynolds](http://faculty.smu.edu/reynolds)
