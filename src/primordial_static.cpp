/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Primordial ODE test problem setup, in a static fluid domain.
 This should give essentially-identical results as primordial_ode,
 but now includes evolution of [stationary] hydrodynamic fields.
 This uses ARKode's MRIStep time-stepping module, via the
 multirate_chem_hydro_main.cpp driver.
---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <dengo_primordial_network.hpp>


// Define global structure for Dengo-based calculations
cvklu_data *network_data;


// Utility routine to initialize global Dengo data structures
int initialize_Dengo_structures(const EulerData& udata) {

  // initialize primordial rate tables, etc
  network_data = NULL;
  network_data = cvklu_setup_data("primordial_tables.h5", NULL, NULL);
  if (network_data == NULL)  return(1);

  // overwrite internal strip size
  network_data->nstrip = udata.nxl * udata.nyl * udata.nzl;

  // set redshift value for non-cosmological run
  network_data->current_z = -1.0;

  return(0);
}


// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const EulerData& udata)
{

  // access data fields
  long int v, i, j, k, idx;
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
    cout << "\nPrimordial static test problem\n\n";

  // ensure that this is compiled with 10 chemical species
  if (udata.nchem != 10) {
    cerr << "\nIncorrect number of chemical fields, exiting\n\n";
    return -1;
  }
  realtype *chem = NULL;
  if (udata.nchem > 0) {
    chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
    if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  }

  // return error if input parameters are inappropriate
  if ((udata.xlbc != BC_NEUMANN) && (udata.xlbc != BC_PERIODIC)) {
    cerr << "\nInappropriate x-left boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.xrbc != BC_NEUMANN) && (udata.xrbc != BC_PERIODIC)) {
    cerr << "\nInappropriate x-right boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.ylbc != BC_NEUMANN) && (udata.ylbc != BC_PERIODIC)) {
    cerr << "\nInappropriate y-left boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.yrbc != BC_NEUMANN) && (udata.yrbc != BC_PERIODIC)) {
    cerr << "\nInappropriate y-right boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.zlbc != BC_NEUMANN) && (udata.zlbc != BC_PERIODIC)) {
    cerr << "\nInappropriate z-left boundary conditions, exiting\n\n";
    return -1;
  }
  if ((udata.zrbc != BC_NEUMANN) && (udata.zrbc != BC_PERIODIC)) {
    cerr << "\nInappropriate z-right boundary conditions, exiting\n\n";
    return -1;
  }

  // ensure that local subdomain size does not exceed dengo 'MAX_NCELLS' preprocessor value
  if (udata.nxl * udata.nyl * udata.nzl > MAX_NCELLS) {
    cerr << "\nTotal spatial subdomain size (" <<
      udata.nxl * udata.nyl * udata.nzl << ") exceeds dengo maximum (" << MAX_NCELLS << ")\n";
    return -1;
  }

  // initial condition values -- essentially-neutral primordial gas
  realtype Tmean = 2000.0;  // mean temperature in K
  realtype Tamp = 1800.0;   // temperature amplitude in K
  realtype tiny = 1e-20;
  realtype mH = 1.67e-24;
  realtype Hfrac = 0.76;
  realtype HI_weight = 1.00794 * mH;
  realtype HII_weight = 1.00794 * mH;
  realtype HM_weight = 1.00794 * mH;
  realtype HeI_weight = 4.002602 * mH;
  realtype HeII_weight = 4.002602 * mH;
  realtype HeIII_weight = 4.002602 * mH;
  realtype H2I_weight = 2*HI_weight;
  realtype H2II_weight = 2*HI_weight;
  realtype gamma = 5.0/3.0;
  realtype kboltz = 1.3806488e-16;
  realtype H2I, H2II, HI, HII, HM, HeI, HeII, HeIII, de, T, ge;
  realtype nH2I, nH2II, nHI, nHII, nHM, nHeI, nHeII, nHeIII, ndens;
  realtype m_amu = 1.66053904e-24;
  realtype density = 1e2 * mH;   // in g/cm^{-3}

  // iterate over subdomain, setting initial conditions
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {

        // set mass densities into local variables
        H2I = 1.e-3*density;
        H2II = tiny*density;
        HII = tiny*density;
        HM = tiny*density;
        HeII = tiny*density;
        HeIII = tiny*density;
        HeI = (ONE-Hfrac)*density - HeII - HeIII;
        HI = density - (H2I+H2II+HII+HM+HeI+HeII+HeIII);

        // compute derived number densities
        nH2I   = H2I   / H2I_weight;
        nH2II  = H2II  / H2II_weight;
        nHII   = HII   / HII_weight;
        nHM    = HM    / HM_weight;
        nHeII  = HeII  / HeII_weight;
        nHeIII = HeIII / HeIII_weight;
        nHeI   = HeI   / HeI_weight;
        nHI    = HI    / HI_weight;
        ndens  = nH2I + nH2II + nHII + nHM + nHeII + nHeIII + nHeI + nHI;
        de     = (nHII + nHeII + 2*nHeIII - nHM + nH2II)*mH;

        // set varying temperature throughout domain, and convert to gas energy
        T = Tmean + Tamp/3.0*( 2.0*(i+udata.is-udata.nx/2)/(udata.nx-1) +
                               2.0*(j+udata.js-udata.ny/2)/(udata.ny-1) +
                               2.0*(k+udata.ks-udata.nz/2)/(udata.nz-1) );
        ge = (kboltz * T * ndens) / (density * (gamma - ONE));

        // insert chemical fields into initial condition vector,
        // converting to 'dimensionless' electron number density
        idx = BUFIDX(v,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        chem[idx+0] = nH2I;
        chem[idx+1] = nH2II;
        chem[idx+2] = nHI;
        chem[idx+3] = nHII;
        chem[idx+4] = nHM;
        chem[idx+5] = nHeI;
        chem[idx+6] = nHeII;
        chem[idx+7] = nHeIII;
        chem[idx+8] = de / m_amu;
        chem[idx+9] = ge;

        // hydrodynamic fields share density and energy with chemical network;
        // all velocities are zero.
        idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
        mx[idx]  = ZERO;
        my[idx]  = ZERO;
        mz[idx]  = ZERO;
        rho[idx] = density;
        et[idx]  = ge + 0.5*(mx[idx]*mx[idx] + my[idx]*my[idx] + mz[idx]*mz[idx])/(rho[idx]*rho[idx]);

      }

  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  N_VConst(ZERO, G);
  return 0;
}

// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const EulerData& udata)
{
  // non-root tasks just exit
  if (udata.myid != 0)  return(0);

  // indices to print
  long int i1 = udata.nxl/3;
  long int j1 = udata.nyl/3;
  long int k1 = udata.nzl/3;
  long int idx1 = BUFIDX(0,i1,j1,k1,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
  long int i2 = 2*udata.nxl/3;
  long int j2 = 2*udata.nyl/3;
  long int k2 = 2*udata.nzl/3;
  long int idx2 = BUFIDX(0,i2,j2,k2,udata.nchem,udata.nxl,udata.nyl,udata.nzl);

  // access chemistry N_Vector data
  realtype *chem = NULL;
  chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (output_diagnostics)", 0)) return -1;

  // set some constants
  realtype mH = 1.67e-24;
  realtype HI_weight = 1.00794 * mH;
  realtype HII_weight = 1.00794 * mH;
  realtype HM_weight = 1.00794 * mH;
  realtype HeI_weight = 4.002602 * mH;
  realtype HeII_weight = 4.002602 * mH;
  realtype HeIII_weight = 4.002602 * mH;
  realtype H2I_weight = 2*HI_weight;
  realtype H2II_weight = 2*HI_weight;
  realtype m_amu = 1.66053904e-24;

  // print current time and number of steps
    printf("\nt = %.3e\n", t);

  // print solutions at first location
  printf("  chem[%li,%li,%li]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         i1, j1, k1,
         network_data->scale[0][idx1+0]*chem[idx1+0],
         network_data->scale[0][idx1+1]*chem[idx1+1],
         network_data->scale[0][idx1+2]*chem[idx1+2],
         network_data->scale[0][idx1+3]*chem[idx1+3],
         network_data->scale[0][idx1+4]*chem[idx1+4],
         network_data->scale[0][idx1+5]*chem[idx1+5],
         network_data->scale[0][idx1+6]*chem[idx1+6],
         network_data->scale[0][idx1+7]*chem[idx1+7],
         network_data->scale[0][idx1+8]*chem[idx1+8],
         network_data->scale[0][idx1+9]*chem[idx1+9]);

  // print solutions at second location
  printf("  chem[%li,%li,%li]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         i2, j2, k2,
         network_data->scale[0][idx2+0]*chem[idx2+0],
         network_data->scale[0][idx2+1]*chem[idx2+1],
         network_data->scale[0][idx2+2]*chem[idx2+2],
         network_data->scale[0][idx2+3]*chem[idx2+3],
         network_data->scale[0][idx2+4]*chem[idx2+4],
         network_data->scale[0][idx2+5]*chem[idx2+5],
         network_data->scale[0][idx2+6]*chem[idx2+6],
         network_data->scale[0][idx2+7]*chem[idx2+7],
         network_data->scale[0][idx2+8]*chem[idx2+8],
         network_data->scale[0][idx2+9]*chem[idx2+9]);

  // return with success
  return(0);
}

//---- end of file ----
