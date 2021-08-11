/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Test problem in which a blast wave proceeds across a "clumpy"
 density field of neutral primordial gas.

 The initial background density field is defined to be

    rho(X) = rho0*(1 + \sum_i s_i*exp(-2*(||X-X_i||/r_i)^2)),

 where s_i, r_i and X_i are clump-dependent.  We place these
 throughout the domain by randomly choosing CLUMPS_PER_PROC*nprocs
 overall clumps in the simulation box; while this is based on a
 uniform distribution, no process is guaranteed to have
 CLUMPS_PER_PROC clumps centered within its domain.  We randomly
 choose the clump "radius" r_i to equal a uniformly-distributed
 random number in the interval
 [dx*MIN_CLUMP_RADIUS, dx*MAX_CLUMP_RADIUS].  Finally, we randomly
 choose the clump "strength" s_i to be a uniformly-distributed
 random number in the interval [0, MAX_CLUMP_STRENGTH].  The
 parameters CLUMPS_PER_PROC, MIN_CLUMP_RADIUS, MAX_CLUMP_RADIUS
 and MAX_CLUMP_STRENGTH are #defined below.

 The background temperature is held at a fixed constant, T0, and
 the fluid is initially at rest (all initial velocities are
 identically zero).  The value of T0 is similarly specified by
 a #define below.

 On top of this background state, we add another Gaussian bump
 to both density **and Temperature**:

    rho_S(X) = rho0*B_DENSITY*exp(-2*(||X-B_CENTER||/B_RADIUS)^2)),
    T_S(X)   = T0*B_TEMPERATURE*exp(-2*(||X-B_CENTER||/B_RADIUS)^2)),

 It is this higher-pressure region that initiates the "blast"
 through the domain.  The values of B_DENSITY, B_TEMPERATURE,
 B_RADIUS and B_CENTER are all #defined below.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#ifdef USERAJA
#include <raja_primordial_network.hpp>
#else
#include <dengo_primordial_network.hpp>
#endif
#include <random>

// basic problem definitions
#define  CLUMPS_PER_PROC     10              // on average
#define  MIN_CLUMP_RADIUS    RCONST(3.0)     // in number of cells
#define  MAX_CLUMP_RADIUS    RCONST(6.0)     // in number of cells
#define  MAX_CLUMP_STRENGTH  RCONST(5.0)     // mult. density factor
#define  T0                  RCONST(10.0)    // background temperature
#define  BLAST_DENSITY       RCONST(5.0)     // mult. density factor
#define  BLAST_TEMPERATURE   RCONST(5.0)     // mult. temperature factor
#define  BLAST_RADIUS        RCONST(0.1)     // relative to unit cube
#define  BLAST_CENTER_X      RCONST(0.5)     // relative to unit cube
#define  BLAST_CENTER_Y      RCONST(0.5)     // relative to unit cube
#define  BLAST_CENTER_Z      RCONST(0.5)     // relative to unit cube


// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const EulerData& udata)
{

  // access data fields
  long int i, j, k, idx;
  int retval;
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
    cout << "\nPrimordial blast test problem\n";

  // ensure that this is compiled with 10 chemical species
  if (udata.nchem != 10) {
    cerr << "\nIncorrect number of chemical fields, exiting\n\n";
    return -1;
  }
  realtype *chem = NULL;
  if (udata.nchem > 0) {
    chem = N_VGetArrayPointer(N_VGetSubvector_MPIManyVector(w,5));
    if (check_flag((void *) chem, "N_VGetArrayPointer (initial_conditions)", 0)) return -1;
  }

#ifndef USERAJA
  // ensure that local subdomain size does not exceed dengo 'MAX_NCELLS' preprocessor value
  if (udata.nxl * udata.nyl * udata.nzl > MAX_NCELLS) {
    cerr << "\nTotal spatial subdomain size (" <<
      udata.nxl * udata.nyl * udata.nzl << ") exceeds dengo maximum (" << MAX_NCELLS << ")\n";
    return -1;
  }
#endif

  // root process determines locations, radii and strength of density clumps
  long int nclumps = CLUMPS_PER_PROC*udata.nprocs;
  double *clump_data = (double*) malloc(nclumps * 5 * sizeof(double));
  if (udata.myid == 0) {

    // initialize mersenne twister with seed equal to the number of MPI ranks (for reproducibility)
    std::mt19937_64 gen(udata.nprocs);
    std::uniform_real_distribution<> cx_d(udata.xl, udata.xr);
    std::uniform_real_distribution<> cy_d(udata.yl, udata.yr);
    std::uniform_real_distribution<> cz_d(udata.zl, udata.zr);
    std::uniform_real_distribution<> cr_d(MIN_CLUMP_RADIUS,MAX_CLUMP_RADIUS);
    std::uniform_real_distribution<> cs_d(ZERO, MAX_CLUMP_STRENGTH);

    // fill clump information
    for (i=0; i<nclumps; i++) {

      // global (x,y,z) coordinates for this clump center
      clump_data[5*i+0] = cx_d(gen);
      clump_data[5*i+1] = cy_d(gen);
      clump_data[5*i+2] = cz_d(gen);

      // radius of clump
      clump_data[5*i+3] = cr_d(gen);

      // strength of clump
      clump_data[5*i+4] = cs_d(gen);

    }

  }

  // root process broadcasts clump information
  retval = MPI_Bcast(clump_data, nclumps*5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (initial_conditions)", 3)) return -1;


  // output clump information
  if (udata.myid == 0) {
    cout << "\nInitializing problem with " << nclumps << " clumps\n";
    // for (i=0; i<nclumps; i++)
    //   cout << "   clump " << i << ", center = (" << clump_data[5*i+0] << ","
    //        << clump_data[5*i+1] << "," << clump_data[5*i+2] << "),  \tradius = "
    //        << clump_data[5*i+3] << " cells,  \tstrength = " << clump_data[5*i+4] << std::endl;
    // cout << "\n'Blast' clump:\n"
    //      << "       overdensity = " << BLAST_DENSITY << std::endl
    //      << "   overtemperature = " << BLAST_TEMPERATURE << std::endl
    //      << "            radius = " << BLAST_RADIUS << std::endl
    //      << "            center = " << BLAST_CENTER_X << ", "
    //      << BLAST_CENTER_Y << ", " << BLAST_CENTER_Z << std::endl;
  }


  // initial condition values -- essentially-neutral primordial gas
  realtype tiny = 1e-40;
  realtype small = 1e-12;
  //realtype small = 1e-16;
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
  realtype kboltz = 1.3806488e-16;
  realtype H2I, H2II, HI, HII, HM, HeI, HeII, HeIII, de, T, ge;
  realtype nH2I, nH2II, nHI, nHII, nHM, nHeI, nHeII, nHeIII, ndens;
  realtype m_amu = 1.66053904e-24;
  realtype density0 = 1e2 * mH;   // in g/cm^{-3}
  realtype density, xloc, yloc, zloc, cx, cy, cz, cr, cs, xdist, ydist, zdist, rsq;
  realtype vx0 = 0.0;   // in cm/s
  realtype vy0 = 0.0;
  realtype vz0 = 0.0;

  // iterate over subdomain, setting initial conditions
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {

        // determine cell center
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;

        // determine density in this cell (via loop over clumps)
        density = ONE;
        for (idx=0; idx<nclumps; idx++) {
          cx = clump_data[5*idx+0];
          cy = clump_data[5*idx+1];
          cz = clump_data[5*idx+2];
          cr = clump_data[5*idx+3]*udata.dx;
          cs = clump_data[5*idx+4];
          //xdist = min( abs(xloc-cx), min( abs(xloc-cx+udata.xr), abs(xloc-cx-udata.xr) ) );
          //ydist = min( abs(yloc-cy), min( abs(yloc-cy+udata.yr), abs(yloc-cy-udata.yr) ) );
          //zdist = min( abs(zloc-cz), min( abs(zloc-cz+udata.zr), abs(zloc-cz-udata.zr) ) );
          xdist = abs(xloc-cx);
          ydist = abs(yloc-cy);
          zdist = abs(zloc-cz);
          rsq = xdist*xdist + ydist*ydist + zdist*zdist;
          density += cs*exp(-2.0*rsq/cr/cr);
        }
        density *= density0;

        // add blast clump density
        cx = udata.xl + BLAST_CENTER_X*(udata.xr - udata.xl);
        cy = udata.yl + BLAST_CENTER_Y*(udata.yr - udata.yl);
        cz = udata.zl + BLAST_CENTER_Z*(udata.zr - udata.zl);
        //xdist = min( abs(xloc-cx), min( abs(xloc-cx+udata.xr), abs(xloc-cx-udata.xr) ) );
        //ydist = min( abs(yloc-cy), min( abs(yloc-cy+udata.yr), abs(yloc-cy-udata.yr) ) );
        //zdist = min( abs(zloc-cz), min( abs(zloc-cz+udata.zr), abs(zloc-cz-udata.zr) ) );
        cr = BLAST_RADIUS*min( udata.xr-udata.xl, min(udata.yr-udata.yl, udata.zr-udata.zl));
        cs = density0*BLAST_DENSITY;
        xdist = abs(xloc-cx);
        ydist = abs(yloc-cy);
        zdist = abs(zloc-cz);
        rsq = xdist*xdist + ydist*ydist + zdist*zdist;
        density += cs*exp(-2.0*rsq/cr/cr);

        // set location-dependent temperature
        T = T0;
        cs = T0*(BLAST_TEMPERATURE-ONE);
        T += cs*exp(-2.0*rsq/cr/cr);

        // set initial mass densities into local variables -- blast clump is essentially
        // only HI and HeI, but outside we have trace amounts of other species.
        // H2I   = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // H2II  = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HII   = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        // HM    = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HeII  = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        // HeIII = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        // H2I   = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        // H2II  = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HII   = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HM    = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HeII  = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HeIII = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        H2I   = small*density;
        H2II  = tiny*density;
        HII   = tiny*density;
        HM    = tiny*density;
        HeII  = tiny*density;
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

        // convert temperature to gas energy
        ge = (kboltz * T * ndens) / (density * (udata.gamma - ONE));

        // insert chemical fields into initial condition vector,
        // converting to 'dimensionless' electron number density
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
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
        // however, we must convert to dimensionless units
        idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
        rho[idx] = density/udata.DensityUnits;
        mx[idx]  = vx0*density/udata.MomentumUnits;
        my[idx]  = vy0*density/udata.MomentumUnits;
        mz[idx]  = vz0*density/udata.MomentumUnits;
        et[idx]  = (ge + 0.5*density*(vx0*vx0 + vy0*vy0 + vz0*vz0))/udata.EnergyUnits;

      }

#ifdef RAJA_CUDA
  // ensure that chemistry values are synchronized to device
  N_VCopyToDevice_Raja(N_VGetSubvector_MPIManyVector(w,5));
#endif

  free(clump_data);

  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  // initialize external forces to zero
  N_VConst(ZERO, G);
  return 0;
}

// Utility routine to initialize Dengo data structures
int initialize_Dengo_structures(EulerData& udata) {

  // initialize primordial rate tables, etc
  cvklu_data *network_data = NULL;
#ifdef USERAJA
  network_data = cvklu_setup_data("primordial_tables.h5", udata.nxl * udata.nyl * udata.nzl,
                                  udata.memhelper, -1.0);
#else
  network_data = cvklu_setup_data("primordial_tables.h5", NULL, NULL);

  // overwrite internal strip size
  network_data->nstrip = (udata.nxl * udata.nyl * udata.nzl);
  // set redshift value for non-cosmological run
  network_data->current_z = -1.0;
#endif
  if (network_data == NULL)  return(1);


  // initialize 'scale' and 'inv_scale' to valid values
#ifdef USERAJA
  double *sc = network_data->scale;
  double *isc = network_data->inv_scale;
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,udata.nxl * udata.nyl * udata.nzl * udata.nchem),
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

  // store pointer to network_data in udata, and return
  udata.RxNetData = (void*) network_data;
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
      iscview(k,j,i,l) = ONE / scview(k,j,i,l);
      cview(k,j,i,l) = ONE;
    }
   });
#else
  realtype *sc = network_data->scale[0];
  realtype *isc = network_data->inv_scale[0];
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
  // return with success
  return(0);
}

//---- end of file ----
