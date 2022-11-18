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
#include <raja_primordial_network.hpp>
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

  // each process determines the locations, radii and strength of its density clumps
  long int nclumps = CLUMPS_PER_PROC*27;  // also store all clumps for nearest neighbors in every direction
  double clump_data[5*27*CLUMPS_PER_PROC];
  for (i=0; i<5*27*CLUMPS_PER_PROC; i++)  clump_data[i] = 0.0;

  // initialize mersenne twister with seed equal to the number of MPI ranks (for reproducibility)
  std::mt19937_64 gen(udata.nprocs);
  std::uniform_real_distribution<> cx_d(udata.is*udata.dx + udata.xl, (udata.is+udata.nxl)*udata.dx + udata.xl);
  std::uniform_real_distribution<> cy_d(udata.js*udata.dy + udata.yl, (udata.js+udata.nyl)*udata.dy + udata.yl);
  std::uniform_real_distribution<> cz_d(udata.ks*udata.dz + udata.zl, (udata.ks+udata.nzl)*udata.dz + udata.zl);
  std::uniform_real_distribution<> cr_d(MIN_CLUMP_RADIUS,MAX_CLUMP_RADIUS);
  std::uniform_real_distribution<> cs_d(ZERO, MAX_CLUMP_STRENGTH);

  // fill local clump information
  for (i=0; i<CLUMPS_PER_PROC; i++) {

    // global (x,y,z) coordinates for this clump center
    clump_data[5*i+0] = cx_d(gen);
    clump_data[5*i+1] = cy_d(gen);
    clump_data[5*i+2] = cz_d(gen);

    // radius of clump
    clump_data[5*i+3] = cr_d(gen);

    // strength of clump
    clump_data[5*i+4] = cs_d(gen);

  }

  // communicate with nearest neighbors to fill remainder of clump_data
  MPI_Request req[52];
  MPI_Status stat[52];
  int neighbors[26], nb;
  int dims[3], periods[3], coords[3], nbcoords[3];
  int num_neighbors=0;
  //    determine set of *unique* neighboring ranks
  retval = MPI_Cart_get(udata.comm, 3, dims, periods, coords);
  if (check_flag(&retval, "MPI_Cart_get (initial_conditions)", 3)) return -1;
  for (k=-1; k<=1; k++)
    for (j=-1; j<=1; j++)
      for (i=-1; i<=1; i++) {
        nbcoords[0] = coords[0]+i;
        nbcoords[1] = coords[1]+j;
        nbcoords[2] = coords[2]+k;
        if ((nbcoords[0] < 0) && (udata.xlbc != BC_PERIODIC)) continue;
        if ((nbcoords[1] < 0) && (udata.ylbc != BC_PERIODIC)) continue;
        if ((nbcoords[2] < 0) && (udata.zlbc != BC_PERIODIC)) continue;
        if ((nbcoords[0] >= dims[0]) && (udata.xlbc != BC_PERIODIC)) continue;
        if ((nbcoords[1] >= dims[1]) && (udata.xlbc != BC_PERIODIC)) continue;
        if ((nbcoords[2] >= dims[2]) && (udata.xlbc != BC_PERIODIC)) continue;
        retval = MPI_Cart_rank(udata.comm, nbcoords, &nb);
        if (check_flag(&retval, "MPI_Cart_rank (initial_conditions)", 3)) return -1;
        for (int l=0; l<num_neighbors; l++)  if (nb == neighbors[l]) continue;
        if (nb == udata.myid) continue;
        neighbors[num_neighbors++] = nb;
      }
  //    initiate communications
  for (i=0; i<num_neighbors; i++) {
    retval = MPI_Irecv(&(clump_data[(i+1)*5*CLUMPS_PER_PROC]), 5*CLUMPS_PER_PROC,
                       MPI_SUNREALTYPE, neighbors[i], MPI_ANY_TAG, udata.comm, &(req[2*i]));
    if (check_flag(&retval, "MPI_Irecv (initial_conditions)", 3)) return -1;
    retval = MPI_Isend(clump_data, 5*CLUMPS_PER_PROC, MPI_SUNREALTYPE, neighbors[i],
                       i, udata.comm, &(req[2*i+1]));
    if (check_flag(&retval, "MPI_Isend (initial_conditions)", 3)) return -1;
  }
  //    wait for communications to complete
  retval = MPI_Waitall(2*num_neighbors, req, stat);
  if (check_flag(&retval, "MPI_Waitall (initial_conditions)", 3)) return -1;

  // output clump information
  if (udata.myid == 0)
    cout << "\nInitializing problem with " << CLUMPS_PER_PROC*udata.nprocs << " clumps\n";

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
          xdist = udata.xDistance(xloc,cx);
          ydist = udata.yDistance(yloc,cy);
          zdist = udata.zDistance(zloc,cz);
          rsq = xdist*xdist + ydist*ydist + zdist*zdist;
          density += cs*exp(-2.0*rsq/cr/cr);
        }
        density *= density0;

        // add blast clump density
        cx = udata.xl + BLAST_CENTER_X*(udata.xr - udata.xl);
        cy = udata.yl + BLAST_CENTER_Y*(udata.yr - udata.yl);
        cz = udata.zl + BLAST_CENTER_Z*(udata.zr - udata.zl);
        cr = BLAST_RADIUS*min( udata.xr-udata.xl, min(udata.yr-udata.yl, udata.zr-udata.zl));
        cs = density0*BLAST_DENSITY;
        xdist = udata.xDistance(xloc,cx);
        ydist = udata.yDistance(yloc,cy);
        zdist = udata.zDistance(zloc,cz);
        rsq = xdist*xdist + ydist*ydist + zdist*zdist;
        density += cs*exp(-2.0*rsq/cr/cr);

        // set location-dependent temperature
        T = T0;
        cs = T0*(BLAST_TEMPERATURE-ONE);
        T += cs*exp(-2.0*rsq/cr/cr);

        // set initial mass densities into local variables -- blast clump is essentially
        // only HI and HeI, but outside we have trace amounts of other species.
        H2I   = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        H2II  = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        HII   = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        HM    = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        HeII  = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        HeIII = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        //
        // H2I   = (rsq/cr/cr < 2.0) ? small*density : 1.e-3*density;
        // H2II  = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HII   = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HM    = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HeII  = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        // HeIII = (rsq/cr/cr < 2.0) ? tiny*density  : 1.e-3*density;
        //
        // H2I   = small*density;
        // H2II  = tiny*density;
        // HII   = tiny*density;
        // HM    = tiny*density;
        // HeII  = tiny*density;
        // HeIII = tiny*density;
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

#ifdef USEDEVICE
  // ensure that chemistry values are synchronized to device
  N_VCopyToDevice_Raja(N_VGetSubvector_MPIManyVector(w,5));
#endif

  //free(clump_data);

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

  // start profiler
  int retval = udata.profile[PR_CHEMSETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);

  // initialize primordial rate tables, etc
  cvklu_data *network_data = NULL;
  network_data = cvklu_setup_data(udata.comm, "primordial_tables.h5",
                                  udata.nxl * udata.nyl * udata.nzl,
                                  udata.memhelper, -1.0);
  if (network_data == NULL)  return(1);


  // initialize 'scale' and 'inv_scale' to valid values
  double *sc = network_data->scale;
  double *isc = network_data->inv_scale;
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,udata.nxl * udata.nyl * udata.nzl * udata.nchem),
                           [=] RAJA_DEVICE (long int i) {
    sc[i] = ONE;
    isc[i] = ONE;
  });

  // ensure that newtork_data structure is synchronized between host/device device memory
#ifdef RAJA_CUDA
  cudaDeviceSynchronize();
#elif RAJA_HIP
  hipDeviceSynchronize();
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
  int nchem = udata.nchem;
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale, udata.nzl,
                                              udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > iscview(network_data->inv_scale, udata.nzl,
                                               udata.nyl, udata.nxl, udata.nchem);
#ifdef USEDEVICE
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetDeviceArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#else
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#endif
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
  int nchem = udata.nchem;
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale,
                                              udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#ifdef USEDEVICE
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetDeviceArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#else
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#endif
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
    for (int l=0; l<nchem; l++) {
      cview(k,j,i,l) *= scview(k,j,i,l);
    }
  });
  return(0);
}


// Utility routine to undo a previous call to apply_Dengo_scaling (does not change 'scale')
int unapply_Dengo_scaling(N_Vector w, EulerData& udata)
{
  // access Dengo data structure
  cvklu_data *network_data = (cvklu_data*) udata.RxNetData;

  // update current overall solution using 'network_data->scale' structure
  int nchem = udata.nchem;
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale,
                                              udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#ifdef USEDEVICE
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetDeviceArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#else
  RAJA::View<double, RAJA::Layout<4> > cview(N_VGetArrayPointer(N_VGetSubvector_MPIManyVector(w,5)),
                                             udata.nzl, udata.nyl, udata.nxl, udata.nchem);
#endif
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
    for (int l=0; l<nchem; l++) {
      cview(k,j,i,l) /= scview(k,j,i,l);
    }
   });
  return(0);
}


// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const EulerData& udata)
{
  // return with success
  return(0);
}

//---- end of file ----
