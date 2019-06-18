/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Simple 'smoke test' problem to ensure that things run and a
 constant-valued state is retained.
---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>

// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const UserData& udata)
{
  // iterate over subdomain, setting initial condition
  long int i, j, k;
  realtype xloc, yloc, zloc;
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
        zloc = (udata.ks+j+HALF)*udata.dz + udata.zl;
        rho[IDX(i,j,k,udata.nxl,udata.nyl)] = RCONST(1.0);
        mx[ IDX(i,j,k,udata.nxl,udata.nyl)] = RCONST(0.5);
        my[ IDX(i,j,k,udata.nxl,udata.nyl)] = RCONST(0.5);
        mz[ IDX(i,j,k,udata.nxl,udata.nyl)] = RCONST(0.5);
        et[ IDX(i,j,k,udata.nxl,udata.nyl)] = RCONST(1.0);
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
        zloc = (udata.ks+j+HALF)*udata.dz + udata.zl;
        Grho[IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        Gmx[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        Gmy[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        Gmz[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        Get[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
      }
  return 0;
}

// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const UserData& udata)
{
  return(check_conservation(t, w, udata));
}

//---- end of file ----
