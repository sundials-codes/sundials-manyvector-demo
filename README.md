# SUNDIALS Multirate+ManyVector Demo

[Note: this project is in active development; do not expect anything to run (or even compile) at present.]

This is a SUNDIALS-based demonstration application to assess and demonstrate the large-scale parallel performance of new capabilities that have been added to SUNDIALS in recent years.  Namely:

1. ARKode's new multirate integration module, MRIStep, allowing high-order accurate calculations that subcycle "fast" processes within "slow" ones.

2. SUNDIALS' new MPIManyVector module, that allows extreme flexibility in how a solution "vector" is staged on computational resources.

To run this demo you will need modern C and C++ compilers.  All dependencies (SUNDIALS and SuiteSparse) for the demo are installed in-place using Spack, which is included in this repository.

Steps showing the process to download this demo code, install the relevant dependencies, and build the demo in a Linux or OS X environment are as follows:

```bash
> git clone https://github.com/drreynolds/sundials-manyvector-demo.git
> cd sundials-manyvector-demo
> .spack/bin/spack install sundials +suite-sparse +mpi
> .spack/bin/spack view symlink libs sundials
> .spack/bin/spack view symlink mpi mpi
> make
```

## (Current) Documentation

This test simulates a 3D nonlinear inviscid compressible Euler equation,
<p align="center"><img src="/tex/9507ca7fb40fcf9f401a999539c43b82.svg?invert_in_darkmode&sanitize=true" align=middle width=148.6959507pt height=16.438356pt/></p>
for independent variables
<p align="center"><img src="/tex/cb730ebe968bc6766332e6886b7e66ab.svg?invert_in_darkmode&sanitize=true" align=middle width=227.07922545pt height=17.031940199999998pt/></p>
where the spatial domain is a three-dimensional cube,
<p align="center"><img src="/tex/8feeb58278ab7a349b97f906b71ef3fd.svg?invert_in_darkmode&sanitize=true" align=middle width=210.46034459999998pt height=16.438356pt/></p>
The differential equation is completed using initial condition
<p align="center"><img src="/tex/286c8cf3db13c0c402500c4e46ad4ce5.svg?invert_in_darkmode&sanitize=true" align=middle width=129.27608594999998pt height=16.438356pt/></p>
and face-specific boundary conditions, [xlbc, xrbc] x [ylbc, yrbc] x [zlbc, zrbc], where each may be any one of

* periodic (0),
* homogeneous Neumann (1), or
* homogeneous Dirichlet (2),

under the restriction that if any boundary is set to "periodic" then the opposite face must also indicate a periodic condition.

Here, the 'solution' is given by <img src="/tex/21dcb90c8119f18e732808f7537ca0bf.svg?invert_in_darkmode&sanitize=true" align=middle width=418.92522045pt height=35.5436301pt/>, that corresponds to the density, x,y,z-momentum, and the total energy per unit volume.  The fluxes are given by
<p align="center"><img src="/tex/edde56579807923c3cd19bd5843fec36.svg?invert_in_darkmode&sanitize=true" align=middle width=384.32648264999995pt height=23.5253469pt/></p>
<p align="center"><img src="/tex/b06018efc55bb0d7f41c8280c3044cb7.svg?invert_in_darkmode&sanitize=true" align=middle width=382.5153948pt height=23.9085792pt/></p>
<p align="center"><img src="/tex/bb51d69cabdadf10a0c7ca34efb9a778.svg?invert_in_darkmode&sanitize=true" align=middle width=381.2064663pt height=23.5253469pt/></p>

the external force <img src="/tex/c441e18e502be64ac772003edac839dc.svg?invert_in_darkmode&sanitize=true" align=middle width=52.94748029999999pt height=24.65753399999998pt/> is test-problem-dependent, and the ideal gas equation of state gives
<img src="/tex/4e77e66de46d63bbda82c6eee3bebd62.svg?invert_in_darkmode&sanitize=true" align=middle width=210.24441734999996pt height=28.670654099999997pt/> and
<img src="/tex/c91df912216678aa99e1b60d4e0e4ad7.svg?invert_in_darkmode&sanitize=true" align=middle width=190.94207219999998pt height=26.76175259999998pt/>,
or equivalently,
<img src="/tex/f36b76f5fe4c25019d30f71fa5275425.svg?invert_in_darkmode&sanitize=true" align=middle width=243.89134604999995pt height=27.94539330000001pt/> and
<img src="/tex/c4ea234157eb6cb45144280748d0ac3e.svg?invert_in_darkmode&sanitize=true" align=middle width=195.67730489999997pt height=26.76175259999998pt/>

We have the parameters

* R is the specific ideal gas constant (287.14 J/kg/K).
* <img src="/tex/aa8cfea83e4502fbd685d6c095494147.svg?invert_in_darkmode&sanitize=true" align=middle width=14.102064899999991pt height=14.15524440000002pt/> is the specific heat capacity at constant volume (717.5 J/kg/K),
* <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> is the ratio of specific heats, <img src="/tex/de015188ab92fa7280f672e82ba7e75c.svg?invert_in_darkmode&sanitize=true" align=middle width=113.34245999999997pt height=28.670654099999997pt/> (1.4),

corresponding to air (predominantly an ideal diatomic gas). The speed of sound in the gas is then given by
<p align="center"><img src="/tex/3f5e478e3d3c690cf7f15c2a9ac1fe4d.svg?invert_in_darkmode&sanitize=true" align=middle width=67.10942039999999pt height=39.452455349999994pt/></p>
The fluid variables above are non-dimensionalized; in standard SI units these would be:

* [rho] = kg / m<img src="/tex/b6c5b75bafc8bbc771fa716cb26245ff.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>,

* [vx] = [vy] = [vz] = m/s, which implies that [mx] = [my] = [mz] = kg / m<img src="/tex/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/> / s, and

* [et] = kg / m / s<img src="/tex/e18b24c87a7c52fd294215d16b42a437.svg?invert_in_darkmode&sanitize=true" align=middle width=6.5525476499999895pt height=26.76175259999998pt/>

Note: the above follows the description in section 7.3.1-7.3.3 of https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html

This program solves the problem using a finite volume spatial semi-discretization over a uniform grid of dimensions `nx` x `ny` x `nz`, with fluxes calculated using a 5th-order WENO reconstruction.  The spatial domain uses a 3D domain decomposition approach for parallelism over nprocs MPI processes, with layout `npx` x `npy` x `npz` defined automatically via the `MPI_Dims_create` utility routine.  Each field is stored in its own parallel `N_Vector` object; these are combined together to form the full "solution" vector <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> using the `MPIManyVector` `N_Vector` module.  The resulting initial-value problem is solved using a temporally-adaptive explicit Runge Kutta method from ARKode's ARKStep module.  The solution is output to disk and solution statistics are optionally output to the screen at specified frequencies, and run statistics are printed at the end.

Individual test problems may be uniquely specified through an input file and an auxiliarly source code file that should be linked with this main routine at compile time.

The input file, `input_euler3D.txt`, contains:

* the ratio of specific heats, <img src="/tex/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode&sanitize=true" align=middle width=9.423880949999988pt height=14.15524440000002pt/> -- `gamma`
* spatial domain, <img src="/tex/9432d83304c1eb0dcb05f092d30a767f.svg?invert_in_darkmode&sanitize=true" align=middle width=11.87217899999999pt height=22.465723500000017pt/> -- `xl`, `xr`, `yl`, `yr`, `zl`, `zr`
* time interval, <img src="/tex/bbde6652efaeb60e967ee67be6440eb7.svg?invert_in_darkmode&sanitize=true" align=middle width=46.033257599999985pt height=24.65753399999998pt/> -- `t0`, `tf`
* spatial discretization dimensions -- `nx`, `ny`, `nz`
* boundary condition types -- `xlbc`, `xrbc`, `ylbc`, `yrbc`, `zlbc`, `zrbc`
* number of desired solution outputs -- `nout`

Additionally, this file contains the parameter `showstats`, a nonzero value enables optional output of RMS averages for each field at the same frequency as the solution is output to disk.

The auxiliary source code file must contain two functions:

* the initial condition function <img src="/tex/d3cb4393199b89ca003e78d3486fa147.svg?invert_in_darkmode&sanitize=true" align=middle width=46.837068299999984pt height=24.65753399999998pt/> must have the signature:

```C++
     int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);
```

* the forcing function <img src="/tex/c441e18e502be64ac772003edac839dc.svg?invert_in_darkmode&sanitize=true" align=middle width=52.94748029999999pt height=24.65753399999998pt/> must have the signature

```C++
     int external_forces(const realtype& t, N_Vector G, const UserData& udata);
```


## Authors
[Daniel R. Reynolds](mailto:reynolds@smu.edu)
