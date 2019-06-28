/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Simple linear advection test problem:
    rho(X,t) = rho0 + Amp*sin(2*pi*(x-vx0*t))
    vx(X,t)  = vx0
    vy(X,t)  = 0
    vz(X,t)  = 0
    p(X,t)   = p0

 Note1: since these are specified in terms of primitive variables,
 we convert between primitive and conserved variables for the
 initial conditions and accuracy results.

 Note2: the choice of x-directional advection above is purely
 illustrative; the direction of advection is specified via
 pre-processor directives.  In order or priority: ADVECTION_Z,
 ADVECTION_Y or ADVECTION_X (default).

 Note3: this problem must be run on a problem with integer
 extents in each Cartesian direction.

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
#define rho0 RCONST(1.0)
#define p0   RCONST(1.0)
#define Amp  RCONST(0.1)
#ifdef ADVECTION_X
  #define vx0  RCONST(0.5)
#else
  #define vx0  RCONST(0.0)
#endif
#ifdef ADVECTION_Y
  #define vy0  RCONST(0.5)
#else
  #define vy0  RCONST(0.0)
#endif
#ifdef ADVECTION_Z
  #define vz0  RCONST(0.5)
#else
  #define vz0  RCONST(0.0)
#endif


// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const UserData& udata)
{
  // verify that NVAR has been set up properly
  if (NVAR != 5) {
    cerr << "initial_conditions error: incorrect NVAR (check Makefile settings)";
    return -1;
  }
  
  // iterate over subdomain, setting initial condition
  long int i, j, k;
  realtype xloc, yloc, zloc;
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
  long int nxl = udata.nxl;
  long int nyl = udata.nyl;
  long int nzl = udata.nzl;

  if (udata.myid == 0) {
#ifdef ADVECTION_X
    cout << "\nLinear advection test problem, x-directional propagation\n\n";
#endif
#ifdef ADVECTION_Y
    cout << "\nLinear advection test problem, y-directional propagation\n\n";
#endif
#ifdef ADVECTION_Z
    cout << "\nLinear advection test problem, z-directional propagation\n\n";
#endif
  }

  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;
#ifdef ADVECTION_X
        rho[IDX(i,j,k,nxl,nyl,nzl)] = rho0 + Amp*sin(twopi*(xloc-vx0*t));
#endif
#ifdef ADVECTION_Y
        rho[IDX(i,j,k,nxl,nyl,nzl)] = rho0 + Amp*sin(twopi*(yloc-vy0*t));
#endif
#ifdef ADVECTION_Z
        rho[IDX(i,j,k,nxl,nyl,nzl)] = rho0 + Amp*sin(twopi*(zloc-vz0*t));
#endif
        mx[ IDX(i,j,k,nxl,nyl,nzl)] = vx0*rho[IDX(i,j,k,nxl,nyl,nzl)];
        my[ IDX(i,j,k,nxl,nyl,nzl)] = vy0*rho[IDX(i,j,k,nxl,nyl,nzl)];
        mz[ IDX(i,j,k,nxl,nyl,nzl)] = vz0*rho[IDX(i,j,k,nxl,nyl,nzl)];
        et[ IDX(i,j,k,nxl,nyl,nzl)] = udata.eos_inv(rho[IDX(i,j,k,nxl,nyl,nzl)],
                                                    mx[ IDX(i,j,k,nxl,nyl,nzl)],
                                                    my[ IDX(i,j,k,nxl,nyl,nzl)],
                                                    mz[ IDX(i,j,k,nxl,nyl,nzl)], p0);
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
  realtype xloc, yloc, zloc, rhotrue, mxtrue, mytrue, mztrue, ettrue, err;
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
        rhotrue = rho0 + Amp*sin(twopi*(xloc-vx0*t));
#endif
#ifdef ADVECTION_Y
        rhotrue = rho0 + Amp*sin(twopi*(yloc-vy0*t));
#endif
#ifdef ADVECTION_Z
        rhotrue = rho0 + Amp*sin(twopi*(zloc-vz0*t));
#endif
        err = abs(rhotrue-rho[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[0] = max(errI[0], err);
        errR[0] += err*err;

        mxtrue = rhotrue*vx0;
        err = abs(mxtrue-mx[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[1] = max(errI[1], err);
        errR[1] += err*err;

        mytrue = rhotrue*vy0;
        err = abs(mytrue-my[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[2] = max(errI[2], err);
        errR[2] += err*err;

        mztrue = rhotrue*vz0;
        err = abs(mztrue-mz[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)]);
        errI[3] = max(errI[3], err);
        errR[3] += err*err;

        ettrue = udata.eos_inv(rhotrue, mxtrue, mytrue, mztrue, p0);
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
  return(0);
}

//---- end of file ----
