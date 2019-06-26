/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 "Hurricane" two-dimensional test test problem (see section 3.2.1 of
  L. Pan, J. Li and K. Xu, "A Few Benchmark Test Cases for
  Higher-Order Euler Solvers," Numer. Math. Theor. Meth. Appl.,
  10:711-736, 2017.
    rho(X,t) = rho0
    vx(X,t)  = v0*sin(theta)
    vy(X,t)  = -v0*cos(theta)
    vz(X,t)  = 0
    p(X,t)   = A*rho0^2
 where theta=arctan(y/x).  We perform their "Critical rotation"
 problem via the parameters: A=25, v0=10, rho0=1, over the
 computational domain [-1,1]^3, using homogeneous Neumann
 boundary conditions, to a final time of t=0.045.

 Note1: we may actually perform this test in any one of the x-y,
 y-z, or z-x planes, as specified via the preprocessor directive
 TEST_XY, TEST_YZ or TEST_ZX; these have order of precedence
 TEST_YZ then TEST_ZX then TEST_XY, defaulting to TEST_XY if
 none are specified.

 Note2: since these are specified in terms of primitive variables,
 we convert between primitive and conserved variables for the
 initial conditions and accuracy results.

---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>

// determine problem directionality
#ifdef TEST_YZ
  #undef TEST_XY
  #undef TEST_ZX
#else
  #ifdef TEST_ZX
    #undef TEST_XY
  #else
    #ifndef TEST_XY
      #define TEST_XY
    #endif
  #endif
#endif


// problem specification constants
#define rho0 RCONST(1.0)
#define v0   RCONST(10.0)
#define Amp  RCONST(25.0)


// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const UserData& udata)
{
  // iterate over subdomain, setting initial condition
  long int i, j, k;
  realtype xloc, yloc, zloc, r, costheta, sintheta;
  const realtype halfpi = RCONST(2.0)*atan(RCONST(1.0));
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
#ifdef TEST_XY
    cout << "\nHurricane test problem (xy-plane)\n\n";
#endif
#ifdef TEST_ZX
    cout << "\nHurricane test problem (zx-plane)\n\n";
#endif
#ifdef TEST_YZ
    cout << "\nHurricane test problem (yz-plane)\n\n";
#endif
  }

  // return error if input parameters are inappropriate
  if ((udata.xlbc != 1) || (udata.xrbc != 1)) {
    cerr << "\nInappropriate x-boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.ylbc != 1) || (udata.yrbc != 1)) {
    cerr << "\nInappropriate y-boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.zlbc != 1) || (udata.zrbc != 1)) {
    cerr << "\nInappropriate z-boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.gamma != TWO)) {
    cerr << "\nInappropriate gamma, exiting\n\n";
    return -1;
  }

  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;

#ifdef TEST_XY
        r = SUNRsqrt(xloc*xloc + yloc*yloc);
        if (r == ZERO)  r = 1e-14;  // protect against division by zero
        costheta = xloc/r;
        sintheta = yloc/r;
        mx[ IDX(i,j,k,nxl,nyl,nzl)] = rho0*v0*sintheta;
        my[ IDX(i,j,k,nxl,nyl,nzl)] = -rho0*v0*costheta;
        mz[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
#endif
#ifdef TEST_ZX
        r = SUNRsqrt(zloc*zloc + xloc*xloc);
        if (r == ZERO)  r = 1e-14;  // protect against division by zero
        costheta = zloc/r;
        sintheta = xloc/r;
        mz[ IDX(i,j,k,nxl,nyl,nzl)] = rho0*v0*sintheta;
        mx[ IDX(i,j,k,nxl,nyl,nzl)] = -rho0*v0*costheta;
        my[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
#endif
#ifdef TEST_YZ
        r = SUNRsqrt(yloc*yloc + zloc*zloc);
        if (r == ZERO)  r = 1e-14;  // protect against division by zero
        costheta = yloc/r;
        sintheta = zloc/r;
        my[ IDX(i,j,k,nxl,nyl,nzl)] = rho0*v0*sintheta;
        mz[ IDX(i,j,k,nxl,nyl,nzl)] = -rho0*v0*costheta;
        mx[ IDX(i,j,k,nxl,nyl,nzl)] = ZERO;
#endif
        rho[IDX(i,j,k,nxl,nyl,nzl)] = rho0;
        et[ IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rho[IDX(i,j,k,nxl,nyl,nzl)],
                                                    mx[ IDX(i,j,k,nxl,nyl,nzl)],
                                                    my[ IDX(i,j,k,nxl,nyl,nzl)],
                                                    mz[ IDX(i,j,k,nxl,nyl,nzl)],
                                                    Amp*pow(rho0,udata.gamma));
      }
  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const UserData& udata)
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

// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const UserData& udata)
{
  // iterate over subdomain, computing solution error
  long int v, i, j, k;
  int retval;
  realtype xloc, yloc, zloc, r, costheta, sintheta, p0prime,
    rthresh, rhotrue, mxtrue, mytrue, mztrue, err;
  realtype errI[] = {ZERO, ZERO, ZERO, ZERO};
  realtype errR[] = {ZERO, ZERO, ZERO, ZERO};
  realtype toterrI[4], toterrR[4];
  const realtype halfpi = RCONST(2.0)*atan(RCONST(1.0));
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (output_diagnostics)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (output_diagnostics)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (output_diagnostics)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (output_diagnostics)", 0)) return -1;

  // set some reusable constants (protect t against division-by-zero)
  p0prime = Amp*udata.gamma*pow(rho0,udata.gamma-ONE);
  rthresh = TWO*t*SUNRsqrt(p0prime);

  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;

#ifdef TEST_XY
        r = SUNRsqrt(xloc*xloc + yloc*yloc);
        if (r == ZERO)  r = 1e-14;  // protect against division by zero
        costheta = xloc/r;
        sintheta = yloc/r;
        if (r < rthresh) {
          rhotrue = r * r / (RCONST(8.0) * Amp * t * t);
          mxtrue = rhotrue * (xloc + yloc) / (TWO * t);
          mytrue = rhotrue * (yloc - xloc) / (TWO * t);
          mztrue = ZERO;
        } else {
          rhotrue = rho0;
          mxtrue = rho0 * ( TWO*t*p0prime*costheta +
                            SUNRsqrt(TWO*p0prime)*SUNRsqrt(r*r-TWO*t*t*p0prime)*sintheta )/r;
          mytrue = rho0 * ( TWO*t*p0prime*sintheta -
                            SUNRsqrt(TWO*p0prime)*SUNRsqrt(r*r-TWO*t*t*p0prime)*costheta )/r;
          mztrue = ZERO;
        }
#endif
#ifdef TEST_ZX
        r = SUNRsqrt(zloc*zloc + xloc*xloc);
        if (r == ZERO)  r = 1e-14;  // protect against division by zero
        costheta = zloc/r;
        sintheta = xloc/r;
        if (r < rthresh) {
          rhotrue = r * r / (RCONST(8.0) * Amp * t * t);
          mztrue = rhotrue * (zloc + xloc) / (TWO * t);
          mxtrue = rhotrue * (xloc - zloc) / (TWO * t);
          mytrue = ZERO;
        } else {
          rhotrue = rho0;
          mztrue = rho0 * ( TWO*t*p0prime*costheta +
                            SUNRsqrt(TWO*p0prime)*SUNRsqrt(r*r-TWO*t*t*p0prime)*sintheta )/r;
          mxtrue = rho0 * ( TWO*t*p0prime*sintheta -
                            SUNRsqrt(TWO*p0prime)*SUNRsqrt(r*r-TWO*t*t*p0prime)*costheta )/r;
          mytrue = ZERO;
        }
#endif
#ifdef TEST_YZ
        r = SUNRsqrt(yloc*yloc+ zloc*zloc);
        if (r == ZERO)  r = 1e-14;  // protect against division by zero
        costheta = yloc/r;
        sintheta = zloc/r;
        if (r < rthresh) {
          rhotrue = r * r / (RCONST(8.0) * Amp * t * t);
          mytrue = rhotrue * (yloc + zloc) / (TWO * t);
          mztrue = rhotrue * (zloc - yloc) / (TWO * t);
          mxtrue = ZERO;
        } else {
          rhotrue = rho0;
          mytrue = rho0 * ( TWO*t*p0prime*costheta +
                            SUNRsqrt(TWO*p0prime)*SUNRsqrt(r*r-TWO*t*t*p0prime)*sintheta )/r;
          mztrue = rho0 * ( TWO*t*p0prime*sintheta -
                            SUNRsqrt(TWO*p0prime)*SUNRsqrt(r*r-TWO*t*t*p0prime)*costheta )/r;
          mxtrue = ZERO;
        }
#endif

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

      }
  retval = MPI_Reduce(errI, toterrI, 4, MPI_SUNREALTYPE, MPI_MAX, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (output_diagnostics)", 3)) return(1);
  retval = MPI_Reduce(errR, toterrR, 4, MPI_SUNREALTYPE, MPI_SUM, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (output_diagnostics)", 3)) return(1);
  for (v=0; v<4; v++)  toterrR[v] = SUNRsqrt(toterrR[v]/udata.nx/udata.ny/udata.nz);
  if (udata.myid == 0) {
    printf("     errI = %9.2e  %9.2e  %9.2e  %9.2e\n", toterrI[0], toterrI[1], toterrI[2], toterrI[3]);
    printf("     errR = %9.2e  %9.2e  %9.2e  %9.2e\n", toterrR[0], toterrR[1], toterrR[2], toterrR[3]);
  }

  // return with success
  return(0);
}

//---- end of file ----
