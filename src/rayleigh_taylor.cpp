/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Rayleigh-Taylor instability test problem (see section 4.6 of
 R. Liska and B. Wendroff, "Comparison of several difference 
 schemes on 1D and 2D test problems for the Euler equations," 
 SIAM J. Sci. Comput., vol. 25, no. 3, pp 995-1017, 2003.  We 
 use the problem setup described for 'single-mode' test in the 
 ATHENA code, at 
 https://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html:

 Domain: (x,y) in [-0.25, 0.25] x [-0.75,0.75].
 Boundary conditions: periodic in x, homogeneous Neumann in y.
 
 rho(y) = {2, if y > 0
          {1, if y <= 0
 
 Constant gravitational acceleration g = 0.1 is applied in the -y
 direction.

 Pressure is given by the condition of hydrostatic equilibrium:
 
    p = p0 - g*rho*y, 

 where p0 = 2.5, and gamma = 1.4, resulting in a sound speed of 
 3.5 in the low density medium at the interface.

 Initial y-directional velocity perturbation:

    v_y = Amp*[1 + cos(4*pi*x)]*[1 + cos(3*pi*y)]
 
 with Amp = 0.0025.

 Note: since these are specified in terms of primitive variables,
 we convert between primitive and conserved variables for the
 initial conditions and accuracy results.

---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>

// problem specification constants
#define rhoTop RCONST(2.0)
#define rhoBot RCONST(1.0)
#define grav   RCONST(0.1)
#define p0     RCONST(2.5)
//#define Amp    RCONST(0.0025)
#define Amp    RCONST(0.01)

// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const UserData& udata)
{
  
  // access data fields
  long int i, j, k, idx;
  realtype xloc, yloc;
  const realtype pi = RCONST(4.0)*atan(RCONST(1.0));
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;

  // output test problem information
  if (udata.myid == 0) 
    cout << "\nRayleigh-Taylor instability test problem\n";

  // return error if input parameters are inappropriate
  if ((udata.xlbc != BC_PERIODIC) || (udata.xrbc != BC_PERIODIC)) {
    cerr << "\nInappropriate x-boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.ylbc != BC_REFLECTING) || (udata.yrbc != BC_REFLECTING)) {
    cerr << "\nInappropriate y-boundary conditions, exiting\n\n";
    return -1;
  }
  if (abs(udata.xl + RCONST(0.25)) > 1e-14 || abs(udata.xr - RCONST(0.25)) > 1e-14) {
    cerr << "\nInappropriate x spatial extents, exiting\n\n";
    return -1;
  }
  if (abs(udata.yl + RCONST(0.75)) > 1e-14 || abs(udata.yr - RCONST(0.75)) > 1e-14) {
    cerr << "\nInappropriate y spatial extents, exiting\n\n";
    return -1;
  }

  // iterate over subdomain, setting initial conditions
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;

        rho[idx] = (yloc > ZERO) ? rhoTop : rhoBot;
        mx[idx]  = ZERO;
        my[idx]  = rho[idx]*Amp*(ONE + cos(RCONST(4.0)*pi*xloc))*(ONE + cos(RCONST(3.0)*pi*yloc));
        mz[idx]  = ZERO;
        et[idx]  = udata.eos_inv(rho[idx], mx[idx], my[idx], mz[idx],
                                 p0 - grav*rho[idx]*yloc);

      }
  
  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const UserData& udata)
{
  // iterate over subdomain, applying external forces
  long int i, j, k;
  realtype *Gmy = N_VGetSubvectorArrayPointer_MPIManyVector(G,2);
  if (check_flag((void *) Gmy, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) 
        Gmy[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)] = -grav;
  return 0;
}

// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const UserData& udata)
{
  // return with success
  return(0);
}

//---- end of file ----
