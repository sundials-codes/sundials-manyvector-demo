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
#ifdef USERAJA
#include <raja_primordial_network.hpp>
#else
#include <dengo_primordial_network.hpp>
#endif


// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const EulerData& udata)
{

  // access data fields
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

#ifndef USERAJA
  // ensure that local subdomain size does not exceed dengo 'MAX_NCELLS' preprocessor value
  if (udata.nxl * udata.nyl * udata.nzl > MAX_NCELLS) {
    cerr << "\nTotal spatial subdomain size (" <<
      udata.nxl * udata.nyl * udata.nzl << ") exceeds dengo maximum (" << MAX_NCELLS << ")\n";
    return -1;
  }
#endif

  // initial condition values -- essentially-neutral primordial gas
  const realtype Tmean = 2000.0;  // mean temperature in K
  const realtype Tamp = 0.0;      // temperature amplitude in K
  const realtype tiny = 1e-40;
  const realtype small = 1e-12;
  //realtype small = 1e-16;
  const realtype mH = 1.67e-24;
  const realtype Hfrac = 0.76;
  const realtype HI_weight = 1.00794 * mH;
  const realtype HII_weight = 1.00794 * mH;
  const realtype HM_weight = 1.00794 * mH;
  const realtype HeI_weight = 4.002602 * mH;
  const realtype HeII_weight = 4.002602 * mH;
  const realtype HeIII_weight = 4.002602 * mH;
  const realtype H2I_weight = 2*HI_weight;
  const realtype H2II_weight = 2*HI_weight;
  const realtype gamma = 5.0/3.0;
  const realtype kboltz = 1.3806488e-16;
  const realtype m_amu = 1.66053904e-24;
  const realtype density = 1e2 * mH;   // in g/cm^{-3}

  // iterate over subdomain, setting initial conditions
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++) {

        // set mass densities into local variables
        const realtype H2I   = tiny*density;
        const realtype H2II  = tiny*density;
        const realtype HII   = small*density;
        const realtype HM    = tiny*density;
        const realtype HeII  = small*density;
        const realtype HeIII = small*density;
        //
        // const realtype H2I   = 1.e-3*density;
        // const realtype H2II  = tiny*density;
        // const realtype HII   = tiny*density;
        // const realtype HM    = tiny*density;
        // const realtype HeII  = tiny*density;
        // const realtype HeIII = tiny*density;
        const realtype HeI = (ONE-Hfrac)*density - HeII - HeIII;
        const realtype HI = density - (H2I+H2II+HII+HM+HeI+HeII+HeIII);

        // compute derived number densities
        const realtype nH2I   = H2I   / H2I_weight;
        const realtype nH2II  = H2II  / H2II_weight;
        const realtype nHII   = HII   / HII_weight;
        const realtype nHM    = HM    / HM_weight;
        const realtype nHeII  = HeII  / HeII_weight;
        const realtype nHeIII = HeIII / HeIII_weight;
        const realtype nHeI   = HeI   / HeI_weight;
        const realtype nHI    = HI    / HI_weight;
        const realtype ndens  = nH2I + nH2II + nHII + nHM + nHeII + nHeIII + nHeI + nHI;
        const realtype de     = (nHII + nHeII + 2*nHeIII - nHM + nH2II)*mH;

        // set varying temperature throughout domain, and convert to gas energy
        const realtype T = Tmean + ( Tamp*(i+udata.is-udata.nx/2)/(udata.nx-1) +
                                     Tamp*(j+udata.js-udata.ny/2)/(udata.ny-1) +
                                     Tamp*(k+udata.ks-udata.nz/2)/(udata.nz-1) )/3.0;
        const realtype ge = (kboltz * T * ndens) / (density * (gamma - ONE));

        // insert chemical fields into initial condition vector,
        // converting to 'dimensionless' electron number density
        long int idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
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
        // all velocities are zero.  However, we must convert to dimensionless units
        idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
        rho[idx] = density/udata.DensityUnits;
        mx[idx]  = ZERO/udata.MomentumUnits;
        my[idx]  = ZERO/udata.MomentumUnits;
        mz[idx]  = ZERO/udata.MomentumUnits;
        et[idx]  = (ge + 0.5/rho[idx]*(mx[idx]*mx[idx] + my[idx]*my[idx] + mz[idx]*mz[idx]))
                 / udata.EnergyUnits;

      }

#ifdef RAJA_CUDA
  // ensure that chemistry values are synchronized to device
  N_VCopyToDevice_Raja(N_VGetSubvector_MPIManyVector(w,5));
#endif

  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  N_VConst(ZERO, G);
  return 0;
}

// Utility routine to initialize global Dengo data structures
int initialize_Dengo_structures(EulerData& udata) {

  // start profiler
  int retval = udata.profile[PR_CHEMSETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);
  
  // initialize primordial rate tables, etc
  cvklu_data *network_data = NULL;
#ifdef USERAJA
  network_data = cvklu_setup_data(udata.comm, "primordial_tables.h5",
                                  udata.nxl * udata.nyl * udata.nzl,
                                  udata.memhelper, -1.0);
#else
  network_data = cvklu_setup_data("primordial_tables.h5", NULL, NULL);
  network_data->current_z = -1.0;
  network_data->nstrip = (udata.nxl * udata.nyl * udata.nzl);
#endif
  if (network_data == NULL)  return(1);

  // initialize 'scale' and 'inv_scale' to valid values
#ifdef USERAJA
  double *sc = network_data->scale;
  double *isc = network_data->inv_scale;
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0, udata.nxl * udata.nyl * udata.nzl * udata.nchem),
                           [=] RAJA_DEVICE (long int i) {
    sc[i] = ONE;
    isc[i] = ONE;
  });
#else
  for (long int i=0; i< (udata.nxl * udata.nyl * udata.nzl * udata.nchem); i++) {
    network_data->scale[0][i] = ONE;
    network_data->inv_scale[0][i] = ONE;
  }
#endif

  // ensure that newtork_data structure is synchronized between host/device device memory
#ifdef RAJA_CUDA
  cudaDeviceSynchronize();
#elif RAJA_HIP
#error RAJA HIP chemistry interface is currently unimplemented
#endif

  // store pointer to network_data in udata, stop profiler, and return
  udata.RxNetData = (void*) network_data;
  retval = udata.profile[PR_CHEMSETUP].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
  return(0);
}


// Utility routine to free Dengo data structures
void free_Dengo_structures(EulerData& udata) {
  // call utility routine to free contents of Dengo_data structure
  cvklu_free_data(udata.RxNetData, udata.memhelper);
  udata.RxNetData = NULL;
}


// Utility routine to prepare N_Vector solution and Dengo data structures
// for subsequent chemical evolution
int prepare_Dengo_structures(realtype& t, N_Vector w, EulerData& udata)
{
  // access Dengo data structure
  cvklu_data *network_data = (cvklu_data*) udata.RxNetData;

  // move current chemical solution values into 'network_data->scale' structure
#ifdef USERAJA
  int nchem = udata.nchem;
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale, udata.nzl,
                                              udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > iscview(network_data->inv_scale, udata.nzl,
                                               udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetDeviceArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
    for (int l=0; l<nchem; l++) {
      scview(k,j,i,l) = cview(k,j,i,l);
      iscview(k,j,i,l) = ONE / cview(k,j,i,l);
      cview(k,j,i,l) = ONE;
    }
   });
#else
  realtype *sc = network_data->scale[0];
  realtype *isc = network_data->scale[0];
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (prepare_Dengo_structures)", 0)) return -1;
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++)
        for (int l=0; l<udata.nchem; l++) {
          long int idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
          sc[idx] = chem[idx];
          isc[idx] = ONE / sc[idx];
          chem[idx] = ONE;
        }
#endif

  // compute auxiliary values within network_data structure
  setting_up_extra_variables( network_data, udata.nxl*udata.nyl*udata.nzl );

  return(0);
}


// Utility routine to temporarily combine solution & scaling components
// into overall N_Vector solution (does not change 'scale')
int apply_Dengo_scaling(N_Vector w, EulerData& udata)
{
  // access Dengo data structure
  cvklu_data *network_data = (cvklu_data*) udata.RxNetData;

  // update current overall solution using 'network_data->scale' structure
#ifdef USERAJA
  int nchem = udata.nchem;
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale,
                                              udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetDeviceArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
    for (int l=0; l<nchem; l++) {
      cview(k,j,i,l) *= scview(k,j,i,l);
    }
  });
#else
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (prepare_Dengo_structures)", 0)) return -1;
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++)
        for (int l=0; l<udata.nchem; l++) {
          long int idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
          chem[idx] *= network_data->scale[0][idx];
        }
#endif
  return(0);
}


// Utility routine to undo a previous call to apply_Dengo_scaling (does not change 'scale')
int unapply_Dengo_scaling(N_Vector w, EulerData& udata)
{
  // access Dengo data structure
  cvklu_data *network_data = (cvklu_data*) udata.RxNetData;

  // update current overall solution using 'network_data->scale' structure
#ifdef USERAJA
  int nchem = udata.nchem;
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale,
                                              udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetDeviceArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
    for (int l=0; l<nchem; l++) {
      cview(k,j,i,l) /= scview(k,j,i,l);
    }
   });
#else
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (prepare_Dengo_structures)", 0)) return -1;
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++)
        for (int l=0; l<udata.nchem; l++) {
          long int idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
          chem[idx] /= network_data->scale[0][idx];
        }
#endif

  return(0);
}


// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const EulerData& udata)
{
  // non-root tasks just exit
  if (udata.myid != 0)  return(0);

  // indices to print
  int i1 = udata.nxl/3;
  int j1 = udata.nyl/3;
  int k1 = udata.nzl/3;
  int i2 = 2*udata.nxl/3;
  int j2 = 2*udata.nyl/3;
  int k2 = 2*udata.nzl/3;

  // access Dengo data structure
  cvklu_data* network_data = (cvklu_data*) udata.RxNetData;

  // print current time
  printf("\nt = %.3e\n", t);

  // print solutions at first location
#ifdef USERAJA
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale, udata.nzl,
                                              udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetDeviceArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch0a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch0b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch1a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch1b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch2a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch2b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch3a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch3b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch4a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch4b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch5a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch5b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch6a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch6b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch7a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch7b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch8a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch8b(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch9a(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> ch9b(ZERO);
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {

    realtype loc_a = ZERO;
    realtype loc_b = ZERO;
    if ((i == i1) && (j = j1) && (k == k1)) {
      loc_a = ONE;
    }
    if ((i == i2) && (j = j2) && (k == k2)) {
      loc_b = ONE;
    }
    ch0a += loc_a * cview(i,j,i,0) * scview(i,j,i,0);
    ch0b += loc_b * cview(i,j,i,0) * scview(i,j,i,0);
    ch1a += loc_a * cview(i,j,i,1) * scview(i,j,i,1);
    ch1b += loc_b * cview(i,j,i,1) * scview(i,j,i,1);
    ch2a += loc_a * cview(i,j,i,2) * scview(i,j,i,2);
    ch2b += loc_b * cview(i,j,i,2) * scview(i,j,i,2);
    ch3a += loc_a * cview(i,j,i,3) * scview(i,j,i,3);
    ch3b += loc_b * cview(i,j,i,3) * scview(i,j,i,3);
    ch4a += loc_a * cview(i,j,i,4) * scview(i,j,i,4);
    ch4b += loc_b * cview(i,j,i,4) * scview(i,j,i,4);
    ch5a += loc_a * cview(i,j,i,5) * scview(i,j,i,5);
    ch5b += loc_b * cview(i,j,i,5) * scview(i,j,i,5);
    ch6a += loc_a * cview(i,j,i,6) * scview(i,j,i,6);
    ch6b += loc_b * cview(i,j,i,6) * scview(i,j,i,6);
    ch7a += loc_a * cview(i,j,i,7) * scview(i,j,i,7);
    ch7b += loc_b * cview(i,j,i,7) * scview(i,j,i,7);
    ch8a += loc_a * cview(i,j,i,8) * scview(i,j,i,8);
    ch8b += loc_b * cview(i,j,i,8) * scview(i,j,i,8);
    ch9a += loc_a * cview(i,j,i,9) * scview(i,j,i,9);
    ch9b += loc_b * cview(i,j,i,9) * scview(i,j,i,9);
  });
  printf("  chem[%i,%i,%i]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         i1, j1, k1, ch0a.get(), ch1a.get(), ch2a.get(), ch3a.get(), ch4a.get(),
         ch5a.get(), ch6a.get(), ch7a.get(), ch8a.get(), ch9a.get());
  printf("  chem[%i,%i,%i]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         i2, j2, k2, ch0b.get(), ch1b.get(), ch2b.get(), ch3b.get(), ch4b.get(),
         ch5b.get(), ch6b.get(), ch7b.get(), ch8b.get(), ch9b.get());
#else
  long int idx1 = BUFIDX(0,i1,j1,k1,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
  long int idx2 = BUFIDX(0,i2,j2,k2,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (output_diagnostics)", 0)) return -1;
  printf("  chem[%i,%i,%i]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
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
  printf("  chem[%i,%i,%i]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
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
#endif

  // return with success
  return(0);
}

//---- end of file ----
