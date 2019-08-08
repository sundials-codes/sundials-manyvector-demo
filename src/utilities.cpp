/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Implementation file for shared utility routines.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>

#define STSIZE 6

// Routine to compute the Euler ODE RHS function f(t,y).
int fEuler(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // access problem data
  EulerData *udata = (EulerData *) user_data;

  // initialize output to zeros
  N_VConst(ZERO, wdot);

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *rhodot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,0);
  if (check_flag((void *) rhodot, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *mxdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,1);
  if (check_flag((void *) mxdot, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *mydot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,2);
  if (check_flag((void *) mydot, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *mzdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,3);
  if (check_flag((void *) mzdot, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *etdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,4);
  if (check_flag((void *) etdot, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  realtype *chem = NULL;
  realtype *chemdot = NULL;
  if (udata->nchem > 0) {
    chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
    if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
    chemdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,5);
    if (check_flag((void *) chemdot, "N_VGetSubvectorArrayPointer (fEuler)", 0)) return -1;
  }

  // Exchange boundary data with neighbors
  int retval = udata->ExchangeStart(w);
  if (check_flag(&retval, "ExchangeStart (fEuler)", 1)) return -1;

  // Initialize wdot with external forcing terms
  retval = external_forces(t, wdot, *udata);
  if (check_flag(&retval, "external_forces (fEuler)", 1)) return -1;

  // Set shortcut variables
  long int nxl = udata->nxl;
  long int nyl = udata->nyl;
  long int nzl = udata->nzl;
  realtype w1d[STSIZE][NVAR];
  long int v, i, j, k, idx;

  // compute face-centered fluxes over subdomain interior
  for (k=3; k<nzl-2; k++)
    for (j=3; j<nyl-2; j++)
      for (i=3; i<nxl-2; i++) {

        // return with failure on non-positive density, energy or pressure
        // (only check this first time)
        idx = IDX(i,j,k,nxl,nyl,nzl);
        retval = udata->legal_state(rho[idx], mx[idx], my[idx], mz[idx], et[idx]);
        if (check_flag(&retval, "legal_state (fEuler)", 1)) return -1;

        // pack 1D x-directional array of variable shortcuts
        udata->pack1D_x(w1d, rho, mx, my, mz, et, chem, i, j, k);
        // compute flux at lower x-directional face
        idx = BUFIDX(0,i,j,k,NVAR,nxl+1,nyl,nzl);
        face_flux(w1d, 0, &(udata->xflux[idx]), *udata);

        // pack 1D y-directional array of variable shortcuts
        udata->pack1D_y(w1d, rho, mx, my, mz, et, chem, i, j, k);
        // compute flux at lower y-directional face
        idx = BUFIDX(0,i,j,k,NVAR,nxl,nyl+1,nzl);
        face_flux(w1d, 1, &(udata->yflux[idx]), *udata);

        // pack 1D z-directional array of variable shortcuts
        udata->pack1D_z(w1d, rho, mx, my, mz, et, chem, i, j, k);
        // compute flux at lower z-directional face
        idx = BUFIDX(0,i,j,k,NVAR,nxl,nyl,nzl+1);
        face_flux(w1d, 2, &(udata->zflux[idx]), *udata);

      }

  // wait for boundary data to arrive from neighbors
  retval = udata->ExchangeEnd();
  if (check_flag(&retval, "ExchangeEnd (fEuler)", 1)) return -1;

  // computing remaining fluxes over boundary (loop over entire domain, but skip interior)
  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {

        // skip strict interior (already computed)
        if ( (k>2) && (k<nzl-2) && (j>2) && (j<nyl-2) && (i>2) && (i<nxl-2) ) continue;

        // return with failure on non-positive density, energy or pressure
        idx = IDX(i,j,k,nxl,nyl,nzl);
        retval = udata->legal_state(rho[idx], mx[idx], my[idx], mz[idx], et[idx]);
        if (check_flag(&retval, "legal_state (fEuler)", 1)) return -1;

        // x-directional fluxes at "lower" face
        udata->pack1D_x_bdry(w1d, rho, mx, my, mz, et, chem, i, j, k);
        idx = BUFIDX(0,i,j,k,NVAR,nxl+1,nyl,nzl);
        face_flux(w1d, 0, &(udata->xflux[idx]), *udata);

        // x-directional fluxes at "upper" boundary face
        if (i == nxl-1) {
          idx = BUFIDX(0,nxl,j,k,NVAR,nxl+1,nyl,nzl);
          udata->pack1D_x_bdry(w1d, rho, mx, my, mz, et, chem, nxl, j, k);
          face_flux(w1d, 0, &(udata->xflux[idx]), *udata);
        }

        // y-directional fluxes at "lower" face
        idx = BUFIDX(0,i,j,k,NVAR,nxl,nyl+1,nzl);
        udata->pack1D_y_bdry(w1d, rho, mx, my, mz, et, chem, i, j, k);
        face_flux(w1d, 1, &(udata->yflux[idx]), *udata);

        // y-directional fluxes at "upper" boundary face
        if (j == nyl-1) {
          idx = BUFIDX(0,i,nyl,k,NVAR,nxl,nyl+1,nzl);
          udata->pack1D_y_bdry(w1d, rho, mx, my, mz, et, chem, i, nyl, k);
          face_flux(w1d, 1, &(udata->yflux[idx]), *udata);
        }

        // z-directional fluxes at "lower" face
        idx = BUFIDX(0,i,j,k,NVAR,nxl,nyl,nzl+1);
        udata->pack1D_z_bdry(w1d, rho, mx, my, mz, et, chem, i, j, k);
        face_flux(w1d, 2, &(udata->zflux[idx]), *udata);

        // z-directional fluxes at "upper" boundary face
        if (k == nzl-1) {
          idx = BUFIDX(0,i,j,nzl,NVAR,nxl,nyl,nzl+1);
          udata->pack1D_z_bdry(w1d, rho, mx, my, mz, et, chem, i, j, nzl);
          face_flux(w1d, 2, &(udata->zflux[idx]), *udata);
        }

      }

  // iterate over subdomain, updating RHS (source terms were already included)
  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {
        idx = IDX(i,j,k,nxl,nyl,nzl);
        rhodot[idx] -= ( ( udata->xflux[BUFIDX(0,i+1,j,  k,  NVAR,nxl+1,nyl,  nzl  )]
                         - udata->xflux[BUFIDX(0,i,  j,  k,  NVAR,nxl+1,nyl,  nzl  )])/(udata->dx)
                       + ( udata->yflux[BUFIDX(0,i,  j+1,k,  NVAR,nxl,  nyl+1,nzl  )]
                         - udata->yflux[BUFIDX(0,i,  j,  k,  NVAR,nxl,  nyl+1,nzl  )])/(udata->dy)
                       + ( udata->zflux[BUFIDX(0,i,  j,  k+1,NVAR,nxl,  nyl,  nzl+1)]
                         - udata->zflux[BUFIDX(0,i,  j,  k,  NVAR,nxl,  nyl,  nzl+1)])/(udata->dz) );
        mxdot[idx]  -= ( ( udata->xflux[BUFIDX(1,i+1,j,  k,  NVAR,nxl+1,nyl,  nzl  )]
                         - udata->xflux[BUFIDX(1,i,  j,  k,  NVAR,nxl+1,nyl,  nzl  )])/(udata->dx)
                       + ( udata->yflux[BUFIDX(1,i,  j+1,k,  NVAR,nxl,  nyl+1,nzl  )]
                         - udata->yflux[BUFIDX(1,i,  j,  k,  NVAR,nxl,  nyl+1,nzl  )])/(udata->dy)
                       + ( udata->zflux[BUFIDX(1,i,  j,  k+1,NVAR,nxl,  nyl,  nzl+1)]
                         - udata->zflux[BUFIDX(1,i,  j,  k,  NVAR,nxl,  nyl,  nzl+1)])/(udata->dz) );
        mydot[idx]  -= ( ( udata->xflux[BUFIDX(2,i+1,j,  k,  NVAR,nxl+1,nyl,  nzl  )]
                         - udata->xflux[BUFIDX(2,i,  j,  k,  NVAR,nxl+1,nyl,  nzl  )])/(udata->dx)
                       + ( udata->yflux[BUFIDX(2,i,  j+1,k,  NVAR,nxl,  nyl+1,nzl  )]
                         - udata->yflux[BUFIDX(2,i,  j,  k,  NVAR,nxl,  nyl+1,nzl  )])/(udata->dy)
                       + ( udata->zflux[BUFIDX(2,i,  j,  k+1,NVAR,nxl,  nyl,  nzl+1)]
                         - udata->zflux[BUFIDX(2,i,  j,  k,  NVAR,nxl,  nyl,  nzl+1)])/(udata->dz) );
        mzdot[idx]  -= ( ( udata->xflux[BUFIDX(3,i+1,j,  k,  NVAR,nxl+1,nyl,  nzl  )]
                         - udata->xflux[BUFIDX(3,i,  j,  k,  NVAR,nxl+1,nyl,  nzl  )])/(udata->dx)
                       + ( udata->yflux[BUFIDX(3,i,  j+1,k,  NVAR,nxl,  nyl+1,nzl  )]
                         - udata->yflux[BUFIDX(3,i,  j,  k,  NVAR,nxl,  nyl+1,nzl  )])/(udata->dy)
                       + ( udata->zflux[BUFIDX(3,i,  j,  k+1,NVAR,nxl,  nyl,  nzl+1)]
                         - udata->zflux[BUFIDX(3,i,  j,  k,  NVAR,nxl,  nyl,  nzl+1)])/(udata->dz) );
        etdot[idx]  -= ( ( udata->xflux[BUFIDX(4,i+1,j,  k,  NVAR,nxl+1,nyl,  nzl  )]
                         - udata->xflux[BUFIDX(4,i,  j,  k,  NVAR,nxl+1,nyl,  nzl  )])/(udata->dx)
                       + ( udata->yflux[BUFIDX(4,i,  j+1,k,  NVAR,nxl,  nyl+1,nzl  )]
                         - udata->yflux[BUFIDX(4,i,  j,  k,  NVAR,nxl,  nyl+1,nzl  )])/(udata->dy)
                       + ( udata->zflux[BUFIDX(4,i,  j,  k+1,NVAR,nxl,  nyl,  nzl+1)]
                         - udata->zflux[BUFIDX(4,i,  j,  k,  NVAR,nxl,  nyl,  nzl+1)])/(udata->dz) );

        // compute RHS for tracer/chemistry species
        if (udata->nchem > 0) {
          for (v=0; v<udata->nchem; v++)
            chemdot[BUFIDX(v,i,j,k,udata->nchem,nxl,nyl,nzl)] -=
              ( ( udata->xflux[BUFIDX(5+v,i+1,j,  k,  NVAR,nxl+1,nyl,  nzl  )]
                - udata->xflux[BUFIDX(5+v,i,  j,  k,  NVAR,nxl+1,nyl,  nzl  )])/(udata->dx)
              + ( udata->yflux[BUFIDX(5+v,i,  j+1,k,  NVAR,nxl,  nyl+1,nzl  )]
                - udata->yflux[BUFIDX(5+v,i,  j,  k,  NVAR,nxl,  nyl+1,nzl  )])/(udata->dy)
              + ( udata->zflux[BUFIDX(5+v,i,  j,  k+1,NVAR,nxl,  nyl,  nzl+1)]
                - udata->zflux[BUFIDX(5+v,i,  j,  k,  NVAR,nxl,  nyl,  nzl+1)])/(udata->dz) );
        }

      }

  // return with success
  return 0;
}


// given a 6-point stencil of solution values,
//   w(x_{j-3}) w(x_{j-2}) w(x_{j-1}), w(x_j), w(x_{j+1}), w(x_{j+2})
// and the flux direction idir, compute the face-centered flux (f_flux)
// at the center of the stencil, x_{j-1/2}.
//
// The input "idir" handles the directionality for the 1D calculation
//    idir = 0  implies x-directional flux
//    idir = 1  implies y-directional flux
//    idir = 2  implies z-directional flux
//
// This precisely follows the recipe laid out in:
// Chi-Wang Shu (2003) "High-order Finite Difference and Finite Volume WENO
// Schemes and Discontinuous Galerkin Methods for CFD," International Journal of
// Computational Fluid Dynamics, 17:2, 107-118, DOI: 10.1080/1061856031000104851
void face_flux(realtype (&w1d)[6][NVAR], const int& idir,
               realtype* f_face, const EulerData& udata)
{
  // local data
  int i, j;
  realtype rhosqrL, rhosqrR, rhosqrbar, u, v, w, H, qsq, csnd, cinv, cisq, gamm, alpha,
    beta1, beta2, beta3, w1, w2, w3, f1, f2, f3;
  realtype RV[5][5], LV[5][5], p[STSIZE], flux[STSIZE][NVAR],
    fproj[STSIZE-1][NVAR], fs[STSIZE-1][NVAR], ff[NVAR];
  const realtype bc = RCONST(1.083333333333333333333333333333333333333);    // 13/12
  const realtype epsilon = 1e-6;

  // convert state to direction-independent version
  if (idir > 0)
    for (i=0; i<(STSIZE); i++)
      swap(w1d[i][1], w1d[i][1+idir]);

  // compute pressures over stencil
  for (i=0; i<(STSIZE); i++)
    p[i] = udata.eos(w1d[i][0], w1d[i][1], w1d[i][2], w1d[i][3], w1d[i][4]);

  // compute Roe-average state at face:
  //   wbar = [sqrt(rho), sqrt(rho)*vx, sqrt(rho)*vy, sqrt(rho)*vz, (e+p)/sqrt(rho)]
  //          [sqrt(rho), mx/sqrt(rho), my/sqrt(rho), mz/sqrt(rho), (e+p)/sqrt(rho)]
  //   u = wbar_2 / wbar_1
  //   v = wbar_3 / wbar_1
  //   w = wbar_4 / wbar_1
  //   H = wbar_5 / wbar_1
  rhosqrL = SUNRsqrt(w1d[2][0]);
  rhosqrR = SUNRsqrt(w1d[3][0]);
  rhosqrbar = HALF*(rhosqrL + rhosqrR);
  u = HALF*(w1d[2][1]/rhosqrL + w1d[3][1]/rhosqrR)/rhosqrbar;
  v = HALF*(w1d[2][2]/rhosqrL + w1d[3][2]/rhosqrR)/rhosqrbar;
  w = HALF*(w1d[2][3]/rhosqrL + w1d[3][3]/rhosqrR)/rhosqrbar;
  H = HALF*((p[2]+w1d[2][4])/rhosqrL + (p[3]+w1d[3][4])/rhosqrR)/rhosqrbar;

  // compute eigenvectors at face (note: eigenvectors for tracers are just identity)
  qsq = u*u + v*v + w*w;
  gamm = udata.gamma-ONE;
  csnd = gamm*(H - HALF*qsq);
  cinv = ONE/csnd;
  cisq = cinv*cinv;
  for (i=0; i<5; i++)
    for (j=0; j<5; j++) {
      RV[i][j] = ZERO;
      LV[i][j] = ZERO;
    }

  RV[0][0] = ONE;
  RV[0][3] = ONE;
  RV[0][4] = ONE;

  RV[1][0] = u-csnd;
  RV[1][3] = u;
  RV[1][4] = u+csnd;

  RV[2][0] = v;
  RV[2][1] = ONE;
  RV[2][3] = v;
  RV[2][4] = v;

  RV[3][0] = w;
  RV[3][2] = ONE;
  RV[3][3] = w;
  RV[3][4] = w;

  RV[4][0] = H-u*csnd;
  RV[4][1] = v;
  RV[4][2] = w;
  RV[4][3] = HALF*qsq;
  RV[4][4] = H+u*csnd;

  LV[0][0] = HALF*cinv*(u + HALF*gamm*qsq);
  LV[0][1] = -HALF*cinv*(gamm*u + ONE);
  LV[0][2] = -HALF*v*gamm*cinv;
  LV[0][3] = -HALF*w*gamm*cinv;
  LV[0][4] = HALF*gamm*cinv;

  LV[1][0] = -v;
  LV[1][2] = ONE;

  LV[2][0] = -w;
  LV[2][3] = ONE;

  LV[3][0] = -gamm*cinv*(qsq - H);
  LV[3][1] = u*gamm*cinv;
  LV[3][2] = v*gamm*cinv;
  LV[3][3] = w*gamm*cinv;
  LV[3][4] = -gamm*cinv;

  LV[4][0] = -HALF*cinv*(u - HALF*gamm*qsq);
  LV[4][1] = -HALF*cinv*(gamm*u - ONE);
  LV[4][2] = -HALF*v*gamm*cinv;
  LV[4][3] = -HALF*w*gamm*cinv;
  LV[4][4] = HALF*gamm*cinv;


  // compute fluxes and max wave speed over stencil
  alpha = ZERO;
  for (j=0; j<(STSIZE); j++) {
    u = w1d[j][1]/w1d[j][0];                      // u = vx = mx/rho
    flux[j][0] = w1d[j][1];                       // f_rho = rho*u = mx
    flux[j][1] = u*w1d[j][1] + p[j];              // f_mx = rho*u*u + p = mx*u + p
    flux[j][2] = u*w1d[j][2];                     // f_my = rho*v*u = my*u
    flux[j][3] = u*w1d[j][3];                     // f_mz = rho*w*u = mz*u
    flux[j][4] = u*(w1d[j][4] + p[j]);            // f_et = u*(et + p)
    for (i=0; i<udata.nchem; i++)
      flux[j][5+i] = u*w1d[j][5+i];               // f_cj = cj*u, j=0,...,nchem-1
    csnd = SUNRsqrt(udata.gamma*p[j]/w1d[j][0]);  // csnd = sqrt(gamma*p/rho)
    alpha = max(alpha, abs(u)+csnd);
  }


  // fp(x_{i+1/2}):

  //   compute right-shifted Lax-Friedrichs flux over left portion of patch
  for (j=0; j<(STSIZE-1); j++)
    for (i=0; i<(NVAR); i++)
      fs[j][i] = HALF*(flux[j][i] + alpha*w1d[j][i]);

  // compute projected flux for fluid fields (copy tracer fluxes)
  for (j=0; j<(STSIZE-1); j++) {
    for (i=0; i<5; i++)
      fproj[j][i] = LV[i][0]*fs[j][0] + LV[i][1]*fs[j][1] + LV[i][2]*fs[j][2]
                  + LV[i][3]*fs[j][3] + LV[i][4]*fs[j][4];
    for (i=0; i<udata.nchem; i++)  fproj[j][5+i] = fs[j][5+i];
  }

  //   compute WENO signed flux
  for (i=0; i<(NVAR); i++) {
    // smoothness indicators
    beta1 = bc*pow(fproj[2][i] - RCONST(2.0)*fproj[3][i] + fproj[4][i],2)
          + FOURTH*pow(RCONST(3.0)*fproj[2][i] - RCONST(4.0)*fproj[3][i] + fproj[4][i],2);
    beta2 = bc*pow(fproj[1][i] - RCONST(2.0)*fproj[2][i] + fproj[3][i],2)
          + FOURTH*pow(fproj[1][i] - fproj[3][i],2);
    beta3 = bc*pow(fproj[0][i] - RCONST(2.0)*fproj[1][i] + fproj[2][i],2)
          + FOURTH*pow(fproj[0][i] - RCONST(4.0)*fproj[1][i] + RCONST(3.0)*fproj[2][i],2);
    // nonlinear weights
    w1 = RCONST(0.3) / ((epsilon + beta1) * (epsilon + beta1));
    w2 = RCONST(0.6) / ((epsilon + beta2) * (epsilon + beta2));
    w3 = RCONST(0.1) / ((epsilon + beta3) * (epsilon + beta3));
    // flux stencils
    f1 = RCONST(0.3333333333333333333333333333333333333333)*fproj[2][i]
       + RCONST(0.8333333333333333333333333333333333333333)*fproj[3][i]
       - RCONST(0.1666666666666666666666666666666666666667)*fproj[4][i];
    f2 = -RCONST(0.1666666666666666666666666666666666666667)*fproj[1][i]
       + RCONST(0.8333333333333333333333333333333333333333)*fproj[2][i]
       + RCONST(0.3333333333333333333333333333333333333333)*fproj[3][i];
    f3 = RCONST(0.3333333333333333333333333333333333333333)*fproj[0][i]
       - RCONST(1.166666666666666666666666666666666666667)*fproj[1][i]
       + RCONST(1.833333333333333333333333333333333333333)*fproj[2][i];
    // resulting signed flux at face
    ff[i] = (f1*w1 + f2*w2 + f3*w3)/(w1 + w2 + w3);
  }


  // fm(x_{i+1/2}):

  //   compute left-shifted Lax-Friedrichs flux over right portion of patch
  for (j=0; j<(STSIZE-1); j++)
    for (i=0; i<(NVAR); i++)
      fs[j][i] = HALF*(flux[j+1][i] - alpha*w1d[j+1][i]);

  /*// compute projected flux for fluid fields (copy tracer fluxes)*/
  // compute projected flux for fluid fields; treat tracers as another densty
  for (j=0; j<(STSIZE-1); j++) {
    for (i=0; i<5; i++)
      fproj[j][i] = LV[i][0]*fs[j][0] + LV[i][1]*fs[j][1] + LV[i][2]*fs[j][2]
                  + LV[i][3]*fs[j][3] + LV[i][4]*fs[j][4];
    for (i=0; i<udata.nchem; i++)  fproj[j][5+i] = fs[j][5+i];
  }

  //   compute WENO signed fluxes
  for (i=0; i<(NVAR); i++) {
    // smoothness indicators
    beta1 = bc*pow(fproj[2][i] - RCONST(2.0)*fproj[3][i] + fproj[4][i],2)
          + FOURTH*pow(RCONST(3.0)*fproj[2][i] - RCONST(4.0)*fproj[3][i] + fproj[4][i],2);
    beta2 = bc*pow(fproj[1][i] - RCONST(2.0)*fproj[2][i] + fproj[3][i],2)
          + FOURTH*pow(fproj[1][i] - fproj[3][i],2);
    beta3 = bc*pow(fproj[0][i] - RCONST(2.0)*fproj[1][i] + fproj[2][i],2)
          + FOURTH*pow(fproj[0][i] - RCONST(4.0)*fproj[1][i] + RCONST(3.0)*fproj[2][i],2);
    // nonlinear weights
    w1 = RCONST(0.1) / ((epsilon + beta1) * (epsilon + beta1));
    w2 = RCONST(0.6) / ((epsilon + beta2) * (epsilon + beta2));
    w3 = RCONST(0.3) / ((epsilon + beta3) * (epsilon + beta3));
    // flux stencils
    f1 = RCONST(1.833333333333333333333333333333333333333)*fproj[2][i]
       - RCONST(1.166666666666666666666666666666666666667)*fproj[3][i]
       + RCONST(0.3333333333333333333333333333333333333333)*fproj[4][i];
    f2 = RCONST(0.3333333333333333333333333333333333333333)*fproj[1][i]
       + RCONST(0.8333333333333333333333333333333333333333)*fproj[2][i]
       - RCONST(0.1666666666666666666666666666666666666667)*fproj[3][i];
    f3 = -RCONST(0.1666666666666666666666666666666666666667)*fproj[0][i]
       + RCONST(0.8333333333333333333333333333333333333333)*fproj[1][i]
       + RCONST(0.3333333333333333333333333333333333333333)*fproj[2][i];
    // resulting signed flux (add to ff)
    ff[i] += (f1*w1 + f2*w2 + f3*w3)/(w1 + w2 + w3);
  }

  // combine signed fluxes into output, converting back to conserved variables
  for (i=0; i<5; i++)
    f_face[i] = RV[i][0]*ff[0] + RV[i][1]*ff[1] + RV[i][2]*ff[2]
              + RV[i][3]*ff[3] + RV[i][4]*ff[4];
  for (i=0; i<udata.nchem; i++)  f_face[5+i] = ff[5+i];

  // convert fluxes to direction-independent version
  if (idir > 0)
    swap(f_face[1], f_face[1+idir]);

}


// Routine to compute maximum CFL-stable step size
int stability(N_Vector w, realtype t, realtype* dt_stab, void* user_data)
{
  // access problem data
  EulerData *udata = (EulerData *) user_data;

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;

  // iterate over subdomain, computing the maximum local wave speed
  realtype u, p, csnd;
  realtype alpha = ZERO;
  for (long int i=0; i<(udata->nxl)*(udata->nyl)*(udata->nzl); i++) {
    u = max( max( abs(mx[i]/rho[i]), abs(mx[i]/rho[i]) ), abs(mx[i]/rho[i]) );
    p = udata->eos(rho[i], mx[i], my[i], mz[i], et[i]);
    csnd = SUNRsqrt((udata->gamma) * p / rho[i]);
    alpha = max(alpha, abs(u+csnd));
  }

  // determine maximum wave speed over entire domain
  int retval = MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_SUNREALTYPE, MPI_MAX, udata->comm);
  if (check_flag(&retval, "MPI_Alleduce (stability)", 3)) MPI_Abort(udata->comm, 1);

  // compute maximum stable step size
  *dt_stab = (udata->cfl) * min(min(udata->dx, udata->dy), udata->dz) / alpha;

  // return with success
  return(0);
}


// Utility routine to check function return values:
//  opt == 0 means SUNDIALS function allocates memory so check if
//           returned NULL pointer
//  opt == 1 means SUNDIALS function returns a flag so check if
//           flag >= 0
//  opt == 2 means function allocates memory so check if returned
//           NULL pointer
//  opt == 3 means MPI function returns a flag, so check if
//           flag != MPI_SUCCESS
//  opt == 4 corresponds to a check for an illegal state, so check if
//           flag >= 0
int check_flag(const void *flagvalue, const string funcname, const int opt)
{
  int *errflag;

  // Check if SUNDIALS function returned NULL pointer - no memory allocated
  if (opt == 0 && flagvalue == NULL) {
    cerr << "\nSUNDIALS_ERROR: " << funcname << " failed - returned NULL pointer\n\n";
    return 1; }

  // Check if flag < 0
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      cerr << "\nSUNDIALS_ERROR: " << funcname << " failed with flag = " << *errflag << "\n\n";
      return 1;
    }
  }

  // Check if function returned NULL pointer - no memory allocated
  else if (opt == 2 && flagvalue == NULL) {
    cerr << "\nMEMORY_ERROR: " << funcname << " failed - returned NULL pointer\n\n";
    return 1; }

  // Check MPI return value
  else if (opt == 3) {
    errflag = (int *) flagvalue;
    if (*errflag != MPI_SUCCESS) {
      cerr << "\nMPI_ERROR: " << funcname << " failed with flag = " << *errflag << "\n\n";
      return 1;
    }
  }

  // Check for legal state return value
  else if (opt == 4) {
    errflag = (int *) flagvalue;
    if (*errflag != MPI_SUCCESS) {
      cerr << "\nSTATE_ERROR: " << funcname << " failed with flag = " << *errflag;
      if (*errflag == 1)  cerr << "  (illegal density)\n\n";
      if (*errflag == 2)  cerr << "  (illegal energy)\n\n";
      if (*errflag == 3)  cerr << "  (illegal density & energy)\n\n";
      if (*errflag == 4)  cerr << "  (illegal pressure)\n\n";
      if (*errflag == 5)  cerr << "  (illegal density & pressure)\n\n";
      if (*errflag == 6)  cerr << "  (illegal energy & pressure)\n\n";
      if (*errflag == 7)  cerr << "  (illegal density, energy & pressure)\n\n";
      return 1;
    }
  }

  return 0;
}


//---- end of file ----
