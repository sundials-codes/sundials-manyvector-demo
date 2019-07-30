/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Sod shock tube test problem (see section 5.2 of
 J.A. Greenough and W.J. Rider, "A quantitative comparison of
 numerical methods for the compressible Euler equations:
 fifth-order WENO and piecewise-linear Godunov," J. Comput.
 Phys., 196:259-281, 2004)
    [rho, vx, vy, vz, p] = { [1, 0, 0, 0, 1]        if dir < 0.5
                           { [0.125, 0, 0, 0, 0.1]  if dir > 0.5
 where "dir" can be any one of x, y or z, depending on the
 desired directionality of the shock tube problem.   This is
 specified via pre-processor directives.  In order or priority:
 ADVECTION_Z, ADVECTION_Y or ADVECTION_X (default).

 Note1: since these are specified in terms of primitive variables,
 we convert between primitive and conserved variables for the
 initial conditions and accuracy results.

 Note2: this problem should be run with homogeneous Neumann
 boundary conditions in all directions.

---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>

// determine problem directionality
#ifdef ADVECTION_Z
  #undef ADVECTION_X
  #undef ADVECTION_Y
#else
  #ifdef ADVECTION_Y
    #undef ADVECTION_X
    #undef ADVECTION_Z
  #else
    #ifndef ADVECTION_X
      #define ADVECTION_X
    #endif
    #undef ADVECTION_Y
    #undef ADVECTION_Z
  #endif
#endif

// problem specification constants
#define rhoL RCONST(1.0)
#define rhoR RCONST(0.125)
#define pL   RCONST(1.0)
#define pR   RCONST(0.1)
#define uL   RCONST(0.0)
#define uR   RCONST(0.0)
#define HALF RCONST(0.5)
#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)
#define TWO  RCONST(2.0)

// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const EulerData& udata)
{
  // verify that NVAR has been set up properly
  if (NVAR != 5) {
    cerr << "initial_conditions error: incorrect NVAR (check Makefile settings)";
    return -1;
  }

  // iterate over subdomain, setting initial condition
  long int i, j, k;
  realtype xloc, yloc, zloc;
  realtype twopi = RCONST(8.0)*atan(ONE);
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
  long int nxl = udata.nxl;
  long int nyl = udata.nyl;
  long int nzl = udata.nzl;

  if (udata.myid == 0) {
#ifdef ADVECTION_X
    cout << "\nSod shock tube, x-directional propagation\n\n";
#endif
#ifdef ADVECTION_Y
    cout << "\nSod shock tube, y-directional propagation\n\n";
#endif
#ifdef ADVECTION_Z
    cout << "\nSod shock tube, z-directional propagation\n\n";
#endif
  }

  // return error if domain or boundary conditions are inappropriate
#ifdef ADVECTION_X
  if ((udata.xl >= HALF) || (udata.xr <= HALF)) {
    cerr << "\nInappropriate spatial domain, exiting\n\n";
    return -1;
  }
  if ((udata.xlbc != BC_NEUMANN) || (udata.xrbc != BC_NEUMANN)) {
    cerr << "\nInappropriate boundary conditions, exiting\n\n";
    return -1;
  }
#endif
#ifdef ADVECTION_Y
  if ((udata.yl >= HALF) || (udata.yr <= HALF)) {
    cerr << "\nInappropriate spatial domain, exiting\n\n";
    return -1;
  }
  if ((udata.ylbc != BC_NEUMANN) || (udata.yrbc != BC_NEUMANN)) {
    cerr << "\nInappropriate boundary conditions, exiting\n\n";
    return -1;
  }
#endif
#ifdef ADVECTION_Z
  if ((udata.zl >= HALF) || (udata.zr <= HALF)) {
    cerr << "\nInappropriate spatial domain, exiting\n\n";
    return -1;
  }
  if ((udata.zlbc != BC_NEUMANN) || (udata.zrbc != BC_NEUMANN)) {
    cerr << "\nInappropriate boundary conditions, exiting\n\n";
    return -1;
  }
#endif

  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;
#ifdef ADVECTION_X
        if (xloc < HALF) {
          rho[IDX(i,j,k,nxl,nyl,nzl)] = rhoL;
          et[IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rhoL, uL, ZERO, ZERO, pL);
          mx[ IDX(i,j,k,nxl,nyl,nzl)] = rhoL*uL;
        } else {
          rho[IDX(i,j,k,nxl,nyl,nzl)] = rhoR;
          et[IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rhoR, uR, ZERO, ZERO, pR);
          mx[ IDX(i,j,k,nxl,nyl,nzl)] = rhoR*uR;
        }
        my[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
        mz[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
#endif
#ifdef ADVECTION_Y
        if (yloc < HALF) {
          rho[IDX(i,j,k,nxl,nyl,nzl)] = rhoL;
          et[IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rhoL, ZERO, uL, ZERO, pL);
          my[ IDX(i,j,k,nxl,nyl,nzl)] = rhoL*uL;
        } else {
          rho[IDX(i,j,k,nxl,nyl,nzl)] = rhoR;
          et[IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rhoR, ZERO, uR, ZERO, pR);
          my[ IDX(i,j,k,nxl,nyl,nzl)] = rhoR*uR;
        }
        mx[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
        mz[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
#endif
#ifdef ADVECTION_Z
        if (zloc < HALF) {
          rho[IDX(i,j,k,nxl,nyl,nzl)] = rhoL;
          et[IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rhoL, ZERO, ZERO, uL, pL);
          mz[ IDX(i,j,k,nxl,nyl,nzl)] = rhoL*uL;
        } else {
          rho[IDX(i,j,k,nxl,nyl,nzl)] = rhoR;
          et[IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rhoR, ZERO, ZERO, uR, pR);
          mz[ IDX(i,j,k,nxl,nyl,nzl)] = rhoR*uR;
        }
        mx[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
        my[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
#endif
      }
  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  // iterate over subdomain, applying external forces
  long int i, j, k;
  realtype xloc, yloc, zloc;
  realtype *Grho = N_VGetSubvectorArrayPointer_MPIManyVector(G,0);
  if (check_flag((void *) Grho, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *Gmx = N_VGetSubvectorArrayPointer_MPIManyVector(G,1);
  if (check_flag((void *) Gmx, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *Gmy = N_VGetSubvectorArrayPointer_MPIManyVector(G,2);
  if (check_flag((void *) Gmy, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *Gmz = N_VGetSubvectorArrayPointer_MPIManyVector(G,3);
  if (check_flag((void *) Gmz, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *Get = N_VGetSubvectorArrayPointer_MPIManyVector(G,4);
  if (check_flag((void *) Get, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;
        Grho[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)] = ZERO;
        Gmx[ IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)] = ZERO;
        Gmy[ IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)] = ZERO;
        Gmz[ IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)] = ZERO;
        Get[ IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)] = ZERO;
      }
  return 0;
}


// Utility routine for exact_Riemann function below
realtype fsecant(const realtype& p4, const realtype& p1, const realtype& p5,
           const realtype& rho1, const realtype& rho5, const realtype& gamma)
{

  // local variables
  realtype z, c1, c5, gm1, gp1, g2, fact;

  z = p4/p5 - ONE;
  c1 = SUNRsqrt(gamma*p1/rho1);
  c5 = SUNRsqrt(gamma*p5/rho5);
  gm1 = gamma - ONE;
  gp1 = gamma + ONE;
  g2  = TWO*gamma;
  fact = gm1/g2*(c5/c1)*z/SUNRsqrt(ONE + gp1/g2*z);
  fact = pow(ONE - fact, g2/gm1);

  return(p1*fact - p4);
}

// Exact 1D Riemann problem solver (retrieves domain from EulerData structure),
// based on Fortran code at http://cococubed.asu.edu/codes/riemann/exact_riemann.f
//
// Inputs: (t,x) location for desired solution,
//         xI location of discontinuity at t=0,
//         gamma parameter for gas equation of state
// Outputs: density (rho), velocity (u) and pressure (p) at (t,x)
int exact_Riemann(const realtype& t, const realtype& x, const realtype &xI,
                  const realtype& gamma, realtype& rho, realtype& u, realtype& p)
{

  int itmax, iter;
  realtype rho1, p1, u1, rho3, p3, u3, rho4, p4, u4, rho5, p5, u5;
  realtype p40, p41, f0, f1, eps, z, gm1, gp1, fact, w, c1, c3, c5, xsh, xcd, xft, xhd;

  // begin solution
  if (pL > pR) {
    rho1 = rhoL;
    p1   = pL;
    u1   = uL;
    rho5 = rhoR;
    p5   = pR;
    u5   = uR;
  } else {
    rho1 = rhoR;
    p1   = pR;
    u1   = uR;
    rho5 = rhoL;
    p5   = pL;
    u5   = uL;
  }

  // solve for post-shock pressure by secant method
  //   initial guesses
  p40 = p1;
  p41 = p5;
  f0 = fsecant(p40, p1, p5, rho1, rho5, gamma);
  itmax = 50;
  eps = 1.e-14;
  for (iter=1; iter<=itmax; iter++) {
    f1 = fsecant(p41, p1, p5, rho1, rho5, gamma);
    if (f1 == f0) break;
    p4 = p41 - (p41 - p40) * f1 / (f1 - f0);
    if ((abs(p4 - p41) / abs(p41)) < eps) break;

    p40 = p41;
    p41 = p4;
    f0  = f1;
    if (iter == itmax) {
      cerr << "exact_Riemann iteration failed to converge\n";
      return(1);
    }
  }

  // compute post-shock density and velocity
  z  = (p4 / p5 - ONE);
  c5 = SUNRsqrt(gamma * p5 / rho5);

  gm1 = gamma - ONE;
  gp1 = gamma + ONE;

  fact = SUNRsqrt(ONE + HALF*gp1*z/gamma);

  u4 = c5*z / (gamma*fact);
  rho4 = rho5 * (ONE + HALF*gp1*z/gamma) / (ONE + HALF*gm1*z/gamma);

  // shock speed
  w = c5 * fact;

  // compute values at foot of rarefaction
  p3 = p4;
  u3 = u4;
  rho3 = rho1 * pow(p3/p1, ONE/gamma);

  // compute positions of waves
  if (pL > pR) {
    c1 = SUNRsqrt(gamma*p1/rho1);
    c3 = SUNRsqrt(gamma*p3/rho3);

    xsh = xI + w*t;
    xcd = xI + u3*t;
    xft = xI + (u3 - c3)*t;
    xhd = xI - c1*t;

    // compute solution as a function of position
    if (x < xhd) {
      rho = rho1;
      p   = p1;
      u   = u1;
    } else if (x < xft) {
      u    = TWO/gp1*(c1 + (x - xI)/t);
      fact = ONE - HALF*gm1*u/c1;
      rho  = rho1 * pow(fact, TWO/gm1);
      p    = p1 * pow(fact, TWO*gamma/gm1);
    } else if (x < xcd) {
      rho = rho3;
      p   = p3;
      u   = u3;
    } else if (x < xsh) {
      rho = rho4;
      p   = p4;
      u   = u4;
    } else {
      rho = rho5;
      p   = p5;
      u   = u5;
    }
  }

  // if pR > pL, reverse solution
  if (pR > pL) {
    c1 = SUNRsqrt(gamma*p1/rho1);
    c3 = SUNRsqrt(gamma*p3/rho3);

    xsh = xI - w*t;
    xcd = xI - u3*t;
    xft = xI - (u3 - c3)*t;
    xhd = xI + c1*t;

    // compute solution as a function of position
    if (x < xsh) {
      rho = rho5;
      p   = p5;
      u   = -u5;
    } else if (x < xcd) {
      rho = rho4;
      p   = p4;
      u   = -u4;
    } else if (x < xft) {
      rho = rho3;
      p   = p3;
      u   = -u3;
    } else if (x < xhd) {
      u    = -TWO/gp1*(c1 + (xI - x)/t);
      fact = ONE + HALF*gm1*u/c1;
      rho  = rho1 * pow(fact, TWO/gm1);
      p    = p1 * pow(fact, TWO*gamma/gm1);
    } else {
      rho = rho1;
      p   = p1;
      u   = -u1;
    }
  }

  // return with success
  return(0);
}


// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const EulerData& udata)
{
  // iterate over subdomain, computing solution error
  long int v, i, j, k;
  int retval;
  realtype xloc, yloc, zloc, rhotrue, utrue, ptrue, mxtrue, mytrue, mztrue, ettrue, err;
  realtype errI[] = {ZERO, ZERO, ZERO, ZERO, ZERO};
  realtype errR[] = {ZERO, ZERO, ZERO, ZERO, ZERO};
  realtype toterrI[NVAR], toterrR[NVAR];
  realtype twopi = RCONST(8.0)*atan(RCONST(1.0));
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
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;
#ifdef ADVECTION_X
        retval = exact_Riemann(t, xloc, HALF, udata.gamma, rhotrue, utrue, ptrue);
        if (check_flag(&retval, "exact_Riemann (output_diagnostics)", 1)) return -1;
        mxtrue = rhotrue*utrue;
        mytrue = ZERO;
        mztrue = ZERO;
#endif
#ifdef ADVECTION_Y
        retval = exact_Riemann(t, yloc, HALF, udata.gamma, rhotrue, utrue, ptrue);
        if (check_flag(&retval, "exact_Riemann (output_diagnostics)", 1)) return -1;
        mxtrue = ZERO;
        mytrue = rhotrue*utrue;
        mztrue = ZERO;
#endif
#ifdef ADVECTION_Z
        retval = exact_Riemann(t, zloc, HALF, udata.gamma, rhotrue, utrue, ptrue);
        if (check_flag(&retval, "exact_Riemann (output_diagnostics)", 1)) return -1;
        mxtrue = ZERO;
        mytrue = ZERO;
        mztrue = rhotrue*utrue;
#endif
        ettrue = udata.eos_inv(rhotrue, mxtrue, mytrue, mztrue, ptrue);

        err = abs(rhotrue-rho[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[0] = max(errI[0], err);
        errR[0] += err*err;

        err = abs(mxtrue-mx[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[1] = max(errI[1], err);
        errR[1] += err*err;

        err = abs(mytrue-my[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[2] = max(errI[2], err);
        errR[2] += err*err;

        err = abs(mztrue-mz[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[3] = max(errI[3], err);
        errR[3] += err*err;

        err = abs(ettrue-et[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[4] = max(errI[4], err);
        errR[4] += err*err;

      }
  retval = MPI_Reduce(errI, toterrI, 5, MPI_SUNREALTYPE, MPI_MAX, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (output_diagnostics)", 3)) return(1);
  retval = MPI_Reduce(errR, toterrR, 5, MPI_SUNREALTYPE, MPI_SUM, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (output_diagnostics)", 3)) return(1);
  for (v=0; v<5; v++)  toterrR[v] = SUNRsqrt(toterrR[v]/udata.nx/udata.ny/udata.nz);
  if (udata.myid == 0) {
    printf("     errI = %9.2e  %9.2e  %9.2e  %9.2e  %9.2e\n",
           toterrI[0], toterrI[1], toterrI[2], toterrI[3], toterrI[4]);
    printf("     errR = %9.2e  %9.2e  %9.2e  %9.2e  %9.2e\n",
           toterrR[0], toterrR[1], toterrR[2], toterrR[3], toterrR[4]);
  }

  // return with success
  return(0);
}

//---- end of file ----
