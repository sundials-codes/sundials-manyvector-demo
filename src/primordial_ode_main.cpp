/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Implementation file to test Dengo interface -- note that although
 this uses the EulerData structure, we do not actually create the
 fluid fields, and all spatial domain and MPI information is
 ignored.  We only use this infrastructure to enable simplified
 input of the time interval and ARKODE solver options, and to
 explore 'equilbrium' configurations for a clumpy density field
 with non-uniform temperature field.
 ---------------------------------------------------------------*/

// Header files
//    Physics
#include <random>
#include <euler3D.hpp>
#ifdef USERAJA
#include <raja_primordial_network.hpp>
#else
#include <dengo_primordial_network.hpp>
#endif

//    SUNDIALS
#ifdef USE_CVODE
#include <cvode/cvode.h>
#include <cvode/cvode_ls.h>
#else
#include <arkode/arkode_arkstep.h>
#endif
#include <sunlinsol/sunlinsol_spgmr.h>
#ifdef USEMAGMA
#include <sunmatrix/sunmatrix_magmadense.h>
#include <sunlinsol/sunlinsol_magmadense.h>
#else
#if defined(RAJA_CUDA)
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#elif defined(CVKLU)
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#else
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#endif
#endif

#ifdef DEBUG
#include "fenv.h"
#endif


// basic problem definitions
#define  CLUMPS_PER_PROC     10              // on average
#define  MIN_CLUMP_RADIUS    RCONST(3.0)     // in number of cells
#define  MAX_CLUMP_RADIUS    RCONST(6.0)     // in number of cells
// #define  MAX_CLUMP_STRENGTH  RCONST(10.0)    // mult. density factor
#define  MAX_CLUMP_STRENGTH  RCONST(5.0)    // mult. density factor
#define  T0                  RCONST(10.0)    // background temperature
// #define  BLAST_DENSITY       RCONST(10.0)    // mult. density factor
#define  BLAST_DENSITY       RCONST(5.0)    // mult. density factor
// #define  BLAST_TEMPERATURE   RCONST(5.0)     // mult. temperature factor
#define  BLAST_TEMPERATURE   RCONST(5.0)     // mult. temperature factor
#define  BLAST_RADIUS        RCONST(0.1)     // relative to unit cube
#define  BLAST_CENTER_X      RCONST(0.5)     // relative to unit cube
#define  BLAST_CENTER_Y      RCONST(0.5)     // relative to unit cube
#define  BLAST_CENTER_Z      RCONST(0.5)     // relative to unit cube


// user-provided functions called by the fast integrators
static int frhs(realtype t, N_Vector w, N_Vector wdot, void* user_data);
static int Jrhs(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

// utility function prototypes
void print_info(void *arkode_mem, realtype &t, N_Vector w,
                cvklu_data *network_data, EulerData &udata);


// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem parameters
  long int N, nstrip;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int idense;                    // flag denoting integration type (dense output vs tstop)
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use (disabled here)
  int nprocs;                    // total number of MPI processes
  N_Vector w = NULL;             // empty vectors for storing overall solution, absolute tolerance array
  N_Vector atols = NULL;
  SUNLinearSolver LS = NULL;     // empty linear solver and matrix structures
  SUNMatrix A = NULL;
  void *arkode_mem = NULL;       // empty ARKStep memory structure
  EulerData udata;               // solver data structures
  ARKODEParameters opts;
#if defined(RAJA_CUDA) && !defined(USEMAGMA)
  cusparseHandle_t cusp_handle;
  cusolverSpHandle_t cusol_handle;
#endif

  //--- General Initialization ---//

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // ensure that this is run in serial
  retval = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (check_flag(&retval, "MPI_Comm_size (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (nprocs != 1) {
    if (myid == 0)
      cerr << "primordial_ode error: test can only be run with 1 MPI task ("
           << nprocs << " used)\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // start various code profilers
  retval = udata.profile[PR_SETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  retval = udata.profile[PR_IO].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // read problem and solver parameters from input file / command line
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  realtype dTout = (udata.tf-udata.t0)/(udata.nout);
  retval = udata.profile[PR_IO].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);
  udata.nchem = 10;   // primordial network requires 10 fields:
                      //   H2_1, H2_2, H_1, H_2, H_m0, He_1, He_2, He_3, de, ge

  // set nstrip value to all cells on this process
  nstrip = udata.nxl * udata.nyl * udata.nzl;

#ifndef USERAJA
  // ensure that nxl*nyl*nzl (inputs) <= MAX_NCELLS (dengo preprocessor value)
  if (nstrip > MAX_NCELLS) {
    cerr << "primordial_ode error: total spatial subdomain size (" <<
      nstrip << ") exceeds maximum (" << MAX_NCELLS << ")\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
#endif

  // Output problem setup information
  bool outproc = (udata.myid == 0);
  if (outproc) {
    cout << "\nPrimordial ODE test problem:\n";
    cout << "   spatial domain: [" << udata.xl << ", " << udata.xr << "] x ["
         << udata.yl << ", " << udata.yr << "] x ["
         << udata.zl << ", " << udata.zr << "]\n";
    cout << "   time domain = (" << udata.t0 << ", " << udata.tf << "]\n";
    cout << "   spatial grid: " << udata.nx << " x " << udata.ny << " x "
         << udata.nz << "\n";
#ifdef USERAJA
#ifdef USEMAGMA
    cout << "Executable built using MAGMA block linear solver\n";
#endif
#if defined(RAJA_CUDA)
    cout << "Executable built with RAJA+CUDA support\n";
#elif defined(RAJA_SERIAL)
    cout << "Executable built with RAJA+SERIAL support\n";
#else
    cout << "Executable built with RAJA+HIP support\n";
#endif
#else
    cout << "Executable built without RAJA support\n";
#endif
  }

  // open solver diagnostics output file for writing
  FILE *DFID = NULL;
  if (udata.showstats && outproc) {
    DFID=fopen("diags_primordial_ode.txt","w");
  }

  // initialize primordial rate tables, etc
  retval = udata.profile[PR_CHEMSETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);
#ifdef USERAJA
  // Initialize ReactionNetwork for host/device reaction rate structure.
  ReactionNetwork network_data = cvklu_setup_data(udata.comm, "primordial_tables.h5",
                                                  nstrip, -1.0, nullptr);
  if (!network_data.IsValid())  return(1);
  //    store pointer to network_data in udata
  udata.RxNetData = (void*) &network_data;
#else
  cvklu_data *network_data = cvklu_setup_data("primordial_tables.h5", NULL, NULL);

  //    overwrite internal strip size
  network_data->nstrip = nstrip;
  //    set redshift value for non-cosmological run
  network_data->current_z = -1.0;
  //    store pointer to network_data in udata
  udata.RxNetData = (void*) network_data;
#endif

  //    stop profiler.
  retval = udata.profile[PR_CHEMSETUP].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

  // initialize N_Vector data structures
  N = (udata.nchem)*nstrip;
#ifdef USERAJA
  w = N_VNewManaged_Raja(N, udata.ctx);
  if (check_flag((void *) w, "N_VNewManaged_Raja (main)", 0)) MPI_Abort(udata.comm, 1);
  atols = N_VNewManaged_Raja(N, udata.ctx);
  if (check_flag((void *) atols, "N_VNewManaged_Raja (main)", 0)) MPI_Abort(udata.comm, 1);
#else
  w = N_VNew_Serial(N, udata.ctx);
  if (check_flag((void *) w, "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
  atols = N_VNew_Serial(N, udata.ctx);
  if (check_flag((void *) atols, "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
#endif

  // root process determines locations, radii and strength of density clumps
  long int nclumps = CLUMPS_PER_PROC*udata.nprocs;
  double *clump_data;
#ifdef RAJA_CUDA
  cudaMallocManaged((void**)&(clump_data), nclumps * 5 * sizeof(double));
#elif RAJA_HIP
#error RAJA HIP chemistry interface is currently unimplemented
#else
  clump_data = (double*) malloc(nclumps * 5 * sizeof(double));
#endif
  if (udata.myid == 0) {

    // initialize mersenne twister with seed equal to the number of MPI ranks (for reproducibility)
    std::mt19937_64 gen(udata.nprocs);
    std::uniform_real_distribution<> cx_d(udata.xl, udata.xr);
    std::uniform_real_distribution<> cy_d(udata.yl, udata.yr);
    std::uniform_real_distribution<> cz_d(udata.zl, udata.zr);
    std::uniform_real_distribution<> cr_d(MIN_CLUMP_RADIUS,MAX_CLUMP_RADIUS);
    std::uniform_real_distribution<> cs_d(ZERO, MAX_CLUMP_STRENGTH);

    // fill clump information
    for (long int i=0; i<nclumps; i++) {

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

  // ensure that clump data is synchronized between host/device device memory
  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )

  // output clump information
  if (udata.myid == 0) {
    cout << "\nInitializing problem with " << nclumps << " clumps:\n";
    for (int i=0; i<nclumps; i++)
      cout << "   clump " << i << ", center = (" << clump_data[5*i+0] << ","
           << clump_data[5*i+1] << "," << clump_data[5*i+2] << "),  \tradius = "
           << clump_data[5*i+3] << " cells,  \tstrength = " << clump_data[5*i+4] << std::endl;
    cout << "\n'Blast' clump:\n"
         << "       overdensity = " << BLAST_DENSITY << std::endl
         << "   overtemperature = " << BLAST_TEMPERATURE << std::endl
         << "            radius = " << BLAST_RADIUS << std::endl
         << "            center = " << BLAST_CENTER_X << ", "
         << BLAST_CENTER_Y << ", " << BLAST_CENTER_Z << std::endl;
  }

  // constants
  const realtype tiny = 1e-40;
  const realtype small = 1e-12;
  //const realtype small = 1e-16;
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
  const realtype kboltz = 1.3806488e-16;
  const realtype m_amu = 1.66053904e-24;
  const realtype density0 = 1e2 * mH;   // in g/cm^{-3}
  const realtype dx = udata.dx;
  const realtype dy = udata.dy;
  const realtype dz = udata.dz;
  const realtype xl = udata.xl;
  const realtype xr = udata.xr;
  const realtype yl = udata.yl;
  const realtype yr = udata.yr;
  const realtype zl = udata.zl;
  const realtype zr = udata.zr;
  const realtype gamma = udata.gamma;
  const long int is = udata.is;
  const long int js = udata.js;
  const long int ks = udata.ks;

  // set initial conditions -- essentially-neutral primordial gas
  realtype *wdata = N_VGetDeviceArrayPointer(w);

#ifdef USERAJA
  RAJA::View<double, RAJA::Layout<4> > wview(wdata, udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
#else
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++) {
#endif

        // cell-specific local variables
        realtype density, xloc, yloc, zloc, cx, cy, cz, cr, cs, xdist, ydist, zdist, rsq;
        realtype H2I, H2II, HI, HII, HM, HeI, HeII, HeIII, de, T, ge;
        realtype nH2I, nH2II, nHI, nHII, nHM, nHeI, nHeII, nHeIII, ndens;

        // determine cell center
        xloc = (is+i+HALF)*dx + xl;
        yloc = (js+j+HALF)*dy + yl;
        zloc = (ks+k+HALF)*dz + zl;

        // determine density in this cell (via loop over clumps)
        density = ONE;
        for (long int idx=0; idx<nclumps; idx++) {
          cx = clump_data[5*idx+0];
          cy = clump_data[5*idx+1];
          cz = clump_data[5*idx+2];
          cr = clump_data[5*idx+3]*dx;
          cs = clump_data[5*idx+4];
          //xdist = min( abs(xloc-cx), min( abs(xloc-cx+xr), abs(xloc-cx-xr) ) );
          //ydist = min( abs(yloc-cy), min( abs(yloc-cy+yr), abs(yloc-cy-yr) ) );
          //zdist = min( abs(zloc-cz), min( abs(zloc-cz+zr), abs(zloc-cz-zr) ) );
          xdist = abs(xloc-cx);
          ydist = abs(yloc-cy);
          zdist = abs(zloc-cz);
          rsq = xdist*xdist + ydist*ydist + zdist*zdist;
          density += cs*exp(-2.0*rsq/cr/cr);
        }
        density *= density0;

        // add blast clump density
        cx = xl + BLAST_CENTER_X*(xr - xl);
        cy = yl + BLAST_CENTER_Y*(yr - yl);
        cz = zl + BLAST_CENTER_Z*(zr - zl);
        //xdist = min( abs(xloc-cx), min( abs(xloc-cx+xr), abs(xloc-cx-xr) ) );
        //ydist = min( abs(yloc-cy), min( abs(yloc-cy+yr), abs(yloc-cy-yr) ) );
        //zdist = min( abs(zloc-cz), min( abs(zloc-cz+zr), abs(zloc-cz-zr) ) );
        cr = BLAST_RADIUS*min( xr-xl, min(yr-yl, zr-zl));
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
        HeI   = (ONE-Hfrac)*density - HeII - HeIII;
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
        ge = (kboltz * T * ndens) / (density * (gamma - ONE));

        // copy final results into vector: H2_1, H2_2, H_1, H_2, H_m0, He_1, He_2, He_3, de, ge;
        // converting to 'dimensionless' electron number density
#ifdef USERAJA
        wview(k,j,i,0) = nH2I;
        wview(k,j,i,1) = nH2II;
        wview(k,j,i,2) = nHI;
        wview(k,j,i,3) = nHII;
        wview(k,j,i,4) = nHM;
        wview(k,j,i,5) = nHeI;
        wview(k,j,i,6) = nHeII;
        wview(k,j,i,7) = nHeIII;
        wview(k,j,i,8) = de / m_amu;
        wview(k,j,i,9) = ge;
      });
#else
        long int idx2 = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        wdata[idx2+0] = nH2I;
        wdata[idx2+1] = nH2II;
        wdata[idx2+2] = nHI;
        wdata[idx2+3] = nHII;
        wdata[idx2+4] = nHM;
        wdata[idx2+5] = nHeI;
        wdata[idx2+6] = nHeII;
        wdata[idx2+7] = nHeIII;
        wdata[idx2+8] = de / m_amu;
        wdata[idx2+9] = ge;
      }
#endif

  // set absolute tolerance array
  N_VConst(opts.atol, atols);
  // realtype *atdata = NULL;
  // atdata = N_VGetArrayPointer(atols);
  // for (k=0; k<udata.nzl; k++)
  //   for (j=0; j<udata.nyl; j++)
  //     for (i=0; i<udata.nxl; i++) {
  //       idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
  //       atdata[idx+0] = opts.atol; // H2I
  //       atdata[idx+1] = opts.atol; // H2II
  //       atdata[idx+2] = opts.atol; // HI
  //       atdata[idx+3] = opts.atol; // HII
  //       atdata[idx+4] = opts.atol; // HM
  //       atdata[idx+5] = opts.atol; // HeI
  //       atdata[idx+6] = opts.atol; // HeII
  //       atdata[idx+7] = opts.atol; // HeIII
  //       atdata[idx+8] = opts.atol; // de
  //       atdata[idx+9] = opts.atol; // ge
  //     }

  // move input solution values into 'scale' components of network_data structure
#ifdef USERAJA
  cvklu_data *h_data = network_data.HPtr();
  int nchem = udata.nchem;
  RAJA::View<double, RAJA::Layout<4> > scview(h_data->scale, udata.nzl,
                                              udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > iscview(h_data->inv_scale, udata.nzl,
                                               udata.nyl, udata.nxl, udata.nchem);
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
    for (int l=0; l<nchem; l++) {
      scview(k,j,i,l) = wview(k,j,i,l);
      iscview(k,j,i,l) = ONE / wview(k,j,i,l);
      wview(k,j,i,l) = ONE;
    }
   });

  // compute auxiliary values within network_data structure
  setting_up_extra_variables(network_data, nstrip);
#else
  double *sc = network_data->scale[0];
  double *isc = network_data->inv_scale[0];
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++)
        for (int l=0; l<udata.nchem; l++) {
          long int idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
          sc[idx] = wdata[idx];
          isc[idx] = ONE / wdata[idx];
          wdata[idx] = ONE;
        }

  // compute auxiliary values within network_data structure
  setting_up_extra_variables(network_data, sc, nstrip);
#endif

  // initialize the integrator memory
#ifdef USE_CVODE
  arkode_mem = CVodeCreate(CV_BDF, udata.ctx);
  if (check_flag((void*) arkode_mem, "CVodeCreate (main)", 0)) MPI_Abort(udata.comm, 1);
  retval = CVodeInit(arkode_mem, frhs, udata.t0, w);
  if (check_flag(&retval, "CVodeInit (main)", 1)) MPI_Abort(udata.comm, 1);
#else
  arkode_mem = ARKStepCreate(NULL, frhs, udata.t0, w, udata.ctx);
  if (check_flag((void*) arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);
#endif

  // create matrix and linear solver modules
  if (opts.iterative) {
    LS = SUNLinSol_SPGMR(w, PREC_NONE, opts.maxliters, udata.ctx);
    if(check_flag((void*) LS, "SUNLinSol_SPGMR (main)", 0)) MPI_Abort(udata.comm, 1);
  } else {
#ifdef USEMAGMA
  // Create SUNMatrix for use in linear solves
  A = SUNMatrix_MagmaDenseBlock(nstrip, udata.nchem, udata.nchem, SUNMEMTYPE_DEVICE,
                                udata.memhelper, NULL, udata.ctx);
  if(check_flag((void *)A, "SUNMatrix_MagmaDenseBlock", 0)) return(1);

  // Create the SUNLinearSolver object
  LS = SUNLinSol_MagmaDense(w, A, udata.ctx);
  if(check_flag((void *)LS, "SUNLinSol_MagmaDense", 0)) return(1);
#else
#if defined RAJA_CUDA
    // Initialize cuSOLVER and cuSPARSE handles
    cusparseCreate(&cusp_handle);
    cusolverSpCreate(&cusol_handle);
    A = SUNMatrix_cuSparse_NewBlockCSR(nstrip, udata.nchem, udata.nchem, 64*udata.nchem,
                                       cusp_handle, udata.ctx);
    if(check_flag((void*) A, "SUNMatrix_cuSparse_NewBlockCSR (main)", 0)) MPI_Abort(udata.comm, 1);
    LS = SUNLinSol_cuSolverSp_batchQR(w, A, cusol_handle, udata.ctx);
    if(check_flag((void*) LS, "SUNLinSol_cuSolverSp_batchQR (main)", 0)) MPI_Abort(udata.comm, 1);
    // Set the sparsity pattern to be fixed
    retval = SUNMatrix_cuSparse_SetFixedPattern(A, SUNTRUE);
    if(check_flag(&retval, "SUNMatrix_cuSolverSp_SetFixedPattern (main)", 0)) MPI_Abort(udata.comm, 1);
    // Initialiize the Jacobian with the fixed sparsity pattern
    retval = initialize_sparse_jacobian_cvklu(A, &udata);
    if(check_flag(&retval, "initialize_sparse_jacobian_cvklu (main)", 0)) MPI_Abort(udata.comm, 1);
#elif defined CVKLU
    A  = SUNSparseMatrix(N, N, 64*nstrip, CSR_MAT, udata.ctx);
    if (check_flag((void*) A, "SUNSparseMatrix (main)", 0)) MPI_Abort(udata.comm, 1);
    LS = SUNLinSol_KLU(w, A, udata.ctx);
    if (check_flag((void*) LS, "SUNLinSol_KLU (main)", 0)) MPI_Abort(udata.comm, 1);
#else
    A  = SUNDenseMatrix(N, N, udata.ctx);
    if (check_flag((void*) A, "SUNDenseMatrix (main)", 0)) MPI_Abort(udata.comm, 1);
    LS = SUNLinSol_Dense(w, A, udata.ctx);
    if (check_flag((void*) LS, "SUNLinSol_Dense (main)", 0)) MPI_Abort(udata.comm, 1);
#endif
#endif
  }

  // attach matrix and linear solver to the integrator
#ifdef USE_CVODE
  retval = CVodeSetLinearSolver(arkode_mem, LS, A);
  if (check_flag(&retval, "CVodeSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);
  if (!opts.iterative) {
    retval = CVodeSetJacFn(arkode_mem, Jrhs);
    if (check_flag(&retval, "CVodeSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);
  }
#else
  retval = ARKStepSetLinearSolver(arkode_mem, LS, A);
  if (check_flag(&retval, "ARKStepSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);
  if (!opts.iterative) {
    retval = ARKStepSetJacFn(arkode_mem, Jrhs);
    if (check_flag(&retval, "ARKStepSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);
  }
#endif

  // setup the ARKStep integrator based on inputs

#ifdef USE_CVODE
  //    pass network_udata to user functions
  retval = CVodeSetUserData(arkode_mem, (void *) (&udata));
  if (check_flag(&retval, "CVodeSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set max order
  if (opts.order != 0) {
    retval = CVodeSetMaxOrd(arkode_mem, opts.order);
    if (check_flag(&retval, "CVodeSetMaxOrd (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    set initial time step size
  retval = CVodeSetInitStep(arkode_mem, opts.h0);
  if (check_flag(&retval, "CVodeSetInitStep (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set minimum time step size
  retval = CVodeSetMinStep(arkode_mem, opts.hmin);
  if (check_flag(&retval, "CVodeSetMinStep (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum time step size
  retval = CVodeSetMaxStep(arkode_mem, opts.hmax);
  if (check_flag(&retval, "CVodeSetMaxStep (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum allowed error test failures
  retval = CVodeSetMaxErrTestFails(arkode_mem, opts.maxnef);
  if (check_flag(&retval, "CVodeSetMaxErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum allowed hnil warnings
  retval = CVodeSetMaxHnilWarns(arkode_mem, opts.mxhnil);
  if (check_flag(&retval, "CVodeSetMaxHnilWarns (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum allowed steps
  retval = CVodeSetMaxNumSteps(arkode_mem, opts.mxsteps);
  if (check_flag(&retval, "CVodeSetMaxNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set tolerances
  retval = CVodeSVtolerances(arkode_mem, opts.rtol, atols);
  if (check_flag(&retval, "CVodeSVtolerances (main)", 1)) MPI_Abort(udata.comm, 1);
#else
  //    pass network_udata to user functions
  retval = ARKStepSetUserData(arkode_mem, (void *) (&udata));
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set diagnostics file
  if (udata.showstats && outproc) {
    retval = ARKStepSetDiagnostics(arkode_mem, DFID);
    if (check_flag(&retval, "ARKStepSetDiagnostics (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    set RK order, or specify individual Butcher table -- "order" overrides "itable"
  if (opts.order != 0) {
    retval = ARKStepSetOrder(arkode_mem, opts.order);
    if (check_flag(&retval, "ARKStepSetOrder (main)", 1)) MPI_Abort(udata.comm, 1);
  } else if (opts.itable != ARKODE_DIRK_NONE) {
    retval = ARKStepSetTableNum(arkode_mem, opts.itable, ARKODE_ERK_NONE);
    if (check_flag(&retval, "ARKStepSetTableNum (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    set dense output order
  retval = ARKStepSetDenseOrder(arkode_mem, opts.dense_order);
  if (check_flag(&retval, "ARKStepSetDenseOrder (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set safety factor
  retval = ARKStepSetSafetyFactor(arkode_mem, opts.safety);
  if (check_flag(&retval, "ARKStepSetSafetyFactor (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set error bias
  retval = ARKStepSetErrorBias(arkode_mem, opts.bias);
  if (check_flag(&retval, "ARKStepSetErrorBias (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set step growth factor
  retval = ARKStepSetMaxGrowth(arkode_mem, opts.growth);
  if (check_flag(&retval, "ARKStepSetMaxGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set time step adaptivity method
  realtype adapt_params[] = {opts.k1, opts.k2, opts.k3};
  int idefault = 1;
  if (abs(opts.k1)+abs(opts.k2)+abs(opts.k3) > 0.0)  idefault=0;
  retval = ARKStepSetAdaptivityMethod(arkode_mem, opts.adapt_method, idefault,
                                      opts.pq, adapt_params);
  if (check_flag(&retval, "ARKStepSetAdaptivityMethod (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set first step growth factor
  retval = ARKStepSetMaxFirstGrowth(arkode_mem, opts.etamx1);
  if (check_flag(&retval, "ARKStepSetMaxFirstGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set error failure growth factor
  retval = ARKStepSetMaxEFailGrowth(arkode_mem, opts.etamxf);
  if (check_flag(&retval, "ARKStepSetMaxEFailGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set initial time step size
  retval = ARKStepSetInitStep(arkode_mem, opts.h0);
  if (check_flag(&retval, "ARKStepSetInitStep (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set minimum time step size
  retval = ARKStepSetMinStep(arkode_mem, opts.hmin);
  if (check_flag(&retval, "ARKStepSetMinStep (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum time step size
  retval = ARKStepSetMaxStep(arkode_mem, opts.hmax);
  if (check_flag(&retval, "ARKStepSetMaxStep (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum allowed error test failures
  retval = ARKStepSetMaxErrTestFails(arkode_mem, opts.maxnef);
  if (check_flag(&retval, "ARKStepSetMaxErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum allowed hnil warnings
  retval = ARKStepSetMaxHnilWarns(arkode_mem, opts.mxhnil);
  if (check_flag(&retval, "ARKStepSetMaxHnilWarns (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set maximum allowed steps
  retval = ARKStepSetMaxNumSteps(arkode_mem, opts.mxsteps);
  if (check_flag(&retval, "ARKStepSetMaxNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set tolerances
  retval = ARKStepSVtolerances(arkode_mem, opts.rtol, atols);
  if (check_flag(&retval, "ARKStepSVtolerances (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set implicit predictor method
  retval = ARKStepSetPredictorMethod(arkode_mem, opts.predictor);
  if (check_flag(&retval, "ARKStepSetPredictorMethod (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set max nonlinear iterations
  retval = ARKStepSetMaxNonlinIters(arkode_mem, opts.maxniters);
  if (check_flag(&retval, "ARKStepSetMaxNonlinIters (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set nonlinear tolerance safety factor
  retval = ARKStepSetNonlinConvCoef(arkode_mem, opts.nlconvcoef);
  if (check_flag(&retval, "ARKStepSetNonlinConvCoef (main)", 1)) MPI_Abort(udata.comm, 1);

#endif

  // Initial batch of outputs
  retval = udata.profile[PR_IO].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  //    Output initial conditions to disk (IMPLEMENT LATER, IF DESIRED)

  //    Output problem-specific diagnostic information
#ifdef USERAJA
  print_info(arkode_mem, udata.t0, w, network_data.HPtr(), udata);
#else
  print_info(arkode_mem, udata.t0, w, network_data, udata);
#endif
  retval = udata.profile[PR_IO].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // If (dense_order == -1), use tstop mode
  if (opts.dense_order == -1)
    idense = 0;
  else   // otherwise tell integrator to use dense output
    idense = 1;

  // stop problem setup profiler
  retval = udata.profile[PR_SETUP].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  /* Main time-stepping loop: calls ARKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  retval = udata.profile[PR_SIMUL].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  realtype t = udata.t0;
  realtype tout = udata.t0+dTout;
  for (int iout=restart; iout<restart+udata.nout; iout++) {

#ifdef USE_CVODE
    if (!idense)
      retval = CVodeSetStopTime(arkode_mem, tout);
    retval = CVode(arkode_mem, tout, w, &t, CV_NORMAL);  // call integrator
#else
    if (!idense)
      retval = ARKStepSetStopTime(arkode_mem, tout);
    retval = ARKStepEvolve(arkode_mem, tout, w, &t, ARK_NORMAL);  // call integrator
#endif
    if (retval >= 0) {                                            // successful solve: update output time
      tout = min(tout+dTout, udata.tf);
    } else {                                                      // unsuccessful solve: break
      if (outproc)
	cerr << "Solver failure, stopping integration\n";
      return 1;
    }

    // periodic output of solution/statistics
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

    //    output statistics to stdout
#ifdef USERAJA
    print_info(arkode_mem, t, w, network_data.HPtr(), udata);
#else
    print_info(arkode_mem, t, w, network_data, udata);
#endif

    //    output results to disk
    //    TODO
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  }
  if (udata.showstats && outproc)  fclose(DFID);


  // reconstruct overall solution values, converting back to mass densities
#ifdef USERAJA
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
        wview(k,j,i,0) *= scview(k,j,i,0) * H2I_weight;    // H2I
        wview(k,j,i,1) *= scview(k,j,i,1) * H2II_weight;   // H2II
        wview(k,j,i,2) *= scview(k,j,i,2) * HI_weight;     // HI
        wview(k,j,i,3) *= scview(k,j,i,3) * HII_weight;    // HII
        wview(k,j,i,4) *= scview(k,j,i,4) * HM_weight;     // HM
        wview(k,j,i,5) *= scview(k,j,i,5) * HeI_weight;    // HeI
        wview(k,j,i,6) *= scview(k,j,i,6) * HeII_weight;   // HeII
        wview(k,j,i,7) *= scview(k,j,i,7) * HeIII_weight;  // HeIII
        wview(k,j,i,8) *= scview(k,j,i,8) * m_amu;         // de
        wview(k,j,i,9) *= scview(k,j,i,9);                 // ge
      });
#else
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++) {
        long int idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        wdata[idx] *= sc[idx] * H2I_weight;        // H2I
        wdata[idx+1] *= sc[idx+1] * H2II_weight;   // H2II
        wdata[idx+2] *= sc[idx+2] * HI_weight;     // HI
        wdata[idx+3] *= sc[idx+3] * HII_weight;    // HII
        wdata[idx+4] *= sc[idx+4] * HM_weight;     // HM
        wdata[idx+5] *= sc[idx+5] * HeI_weight;    // HeI
        wdata[idx+6] *= sc[idx+6] * HeII_weight;   // HeII
        wdata[idx+7] *= sc[idx+7] * HeIII_weight;  // HeIII
        wdata[idx+8] *= sc[idx+8] * m_amu;         // de
        wdata[idx+9] *= sc[idx+9];                 // ge
      }
#endif

  // compute simulation time
  retval = udata.profile[PR_SIMUL].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // Print some final statistics
  long int nst, nst_a, nfe, nfi, netf, nni, ncf;
  long int nls, nje, nli, nlcf, nfls;
  nst = nst_a = nfe = nfi = netf = nni = ncf = 0;
  nls = nje = nli = nlcf = nfls = 0;
#ifdef USE_CVODE
  retval = CVodeGetNumSteps(arkode_mem, &nst);
  if (check_flag(&retval, "CVodeGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumRhsEvals(arkode_mem, &nfi);
  if (check_flag(&retval, "CVodeGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(&retval, "CVodeGetNumErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNonlinSolvStats(arkode_mem, &nni, &ncf);
  if (check_flag(&retval, "CVodeGetNonlinSolvStats (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumLinSolvSetups(arkode_mem, &nls);
  if (check_flag(&retval, "CVodeGetNumLinSolvSetups (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumJacEvals(arkode_mem, &nje);
  if (check_flag(&retval, "CVodeGetNumJacEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumLinIters(arkode_mem, &nli);
  if (check_flag(&retval, "CVodeGetNumLinIters (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumLinConvFails(arkode_mem, &nlcf);
  if (check_flag(&retval, "CVodeGetNumLinConvFails (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumLinRhsEvals(arkode_mem, &nfls);
  if (check_flag(&retval, "CVodeGetNumLinRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
#else
  retval = ARKStepGetNumSteps(arkode_mem, &nst);
  if (check_flag(&retval, "ARKStepGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumStepAttempts(arkode_mem, &nst_a);
  if (check_flag(&retval, "ARKStepGetNumStepAttempts (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumRhsEvals(arkode_mem, &nfe, &nfi);
  if (check_flag(&retval, "ARKStepGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(&retval, "ARKStepGetNumErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNonlinSolvStats(arkode_mem, &nni, &ncf);
  if (check_flag(&retval, "ARKStepGetNonlinSolvStats (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinSolvSetups(arkode_mem, &nls);
  if (check_flag(&retval, "ARKStepGetNumLinSolvSetups (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumJacEvals(arkode_mem, &nje);
  if (check_flag(&retval, "ARKStepGetNumJacEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinIters(arkode_mem, &nli);
  if (check_flag(&retval, "ARKStepGetNumLinIters (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinConvFails(arkode_mem, &nlcf);
  if (check_flag(&retval, "ARKStepGetNumLinConvFails (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinRhsEvals(arkode_mem, &nfls);
  if (check_flag(&retval, "ARKStepGetNumLinRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
#endif

  if (outproc) {
    cout << "\nFinal Solver Statistics:\n";
    cout << "   Internal solver steps = " << nst << " (attempted = " << nst_a << ")\n";
    cout << "   Total RHS evals:  Fe = " << nfe << ",  Fi = " << nfi << "\n";
    cout << "   Total number of error test failures = " << netf << "\n";
    if (opts.iterative && nli > 0) {
      cout << "   Total number of lin iters = " << nli << "\n";
      cout << "   Total number of lin conv fails = " << nlcf << "\n";
      cout << "   Total number of lin RHS evals = " << nfls << "\n";
    } else if (nls > 0) {
      cout << "   Total number of lin solv setups = " << nls << "\n";
      cout << "   Total number of Jac evals = " << nje << "\n";
    }
    if (nni > 0) {
      cout << "   Total number of nonlin iters = " << nni << "\n";
      cout << "   Total number of nonlin conv fails = " << ncf << "\n";
    }
    udata.profile[PR_SETUP].print_cumulative_times("setup");
    udata.profile[PR_IO].print_cumulative_times("I/O");
    udata.profile[PR_SIMUL].print_cumulative_times("sim");
  }

  // Clean up and return with successful completion
#ifdef USERAJA
  cvklu_free_data(network_data);  // Free ReactionNetwork data structure
#else
  free(network_data);
#endif
  udata.RxNetData = NULL;
#ifdef RAJA_SERIAL
  free(clump_data);
#elif RAJA_CUDA
  cudaFree(clump_data);
#else
  // RAJA_HIP is not implemented
#endif
  N_VDestroy(w);                  // Free solution and absolute tolerance vectors
  N_VDestroy(atols);
  SUNLinSolFree(LS);              // Free matrix and linear solver
  SUNMatDestroy(A);
#if defined(RAJA_CUDA) && !defined(USEMAGMA)
  if (!opts.iterative) {
    cusparseDestroy(cusp_handle);   // Destroy the cuSOLVER and cuSPARSE handles
    cusolverSpDestroy(cusol_handle);
  }
#endif
#ifdef USE_CVODE
  CVodeFree(&arkode_mem);         // Free integrator memory
#else
  ARKStepFree(&arkode_mem);       // Free integrator memory
#endif
  MPI_Finalize();                 // Finalize MPI
  return 0;
}


//---- problem-defining functions (wrappers for other routines) ----

static int frhs(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // start timer
  EulerData *udata = (EulerData*) user_data;
  int retval = udata->profile[PR_RHSFAST].start();
  if (check_flag(&retval, "Profile::start (frhs)", 1)) return(-1);

  // initialize all outputs to zero (necessary!!)
  N_VConst(ZERO, wdot);

  // call Dengo RHS routine
#ifdef USERAJA
  retval = calculate_rhs_cvklu(t, w, wdot, (udata->nxl)*(udata->nyl)*(udata->nzl),
                               udata->RxNetData);
#else
  retval = calculate_rhs_cvklu(t, w, wdot, udata->RxNetData);
#endif
  if (check_flag(&retval, "calculate_rhs_cvklu (frhs)", 1)) return(retval);

  // stop timer and return
  retval = udata->profile[PR_RHSFAST].stop();
  if (check_flag(&retval, "Profile::stop (frhs)", 1)) return(-1);
  return(0);
}

static int Jrhs(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // start timer
  EulerData *udata = (EulerData*) user_data;
  int retval = udata->profile[PR_JACFAST].start();
  if (check_flag(&retval, "Profile::start (Jrhs)", 1)) return(-1);

  // call Jacobian routine
#ifdef USERAJA
  retval = calculate_jacobian_cvklu(t, w, fw, Jac, (udata->nxl)*(udata->nyl)*(udata->nzl),
                                    udata->RxNetData, tmp1, tmp2, tmp3);
#else
  retval = calculate_sparse_jacobian_cvklu(t, w, fw, Jac, udata->RxNetData,
                                           tmp1, tmp2, tmp3);
#endif
  if (check_flag(&retval, "calculate_jacobian_cvklu (Jrhs)", 1)) return(retval);

  // stop timer and return
  retval = udata->profile[PR_JACFAST].stop();
  if (check_flag(&retval, "Profile::stop (Jrhs)", 1)) return(-1);
  return(0);
}

// Prints out solution statistics over the domain
void print_info(void *arkode_mem, realtype &t, N_Vector w,
                cvklu_data *network_data, EulerData &udata)
{
  // access N_Vector data
  realtype *wdata = N_VGetArrayPointer(w);
  if (wdata == NULL)  return;

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

  // get current number of time steps
  long int nst = 0;
#ifdef USE_CVODE
  CVodeGetNumSteps(arkode_mem, &nst);
#else
  ARKStepGetNumSteps(arkode_mem, &nst);
#endif

  // print current time and number of steps
  printf("\nt = %.3e  (nst = %li)\n", t, nst);

  // determine mean, min, max values for each component
#ifdef USERAJA
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean0(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean1(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean2(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean3(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean4(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean5(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean6(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean7(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean8(ZERO);
  RAJA::ReduceSum<REDUCEPOLICY, double> cmean9(ZERO);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin0(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin1(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin2(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin3(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin4(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin5(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin6(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin7(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin8(1e300);
  RAJA::ReduceMin<REDUCEPOLICY, double> cmin9(1e300);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax0(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax1(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax2(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax3(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax4(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax5(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax6(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax7(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax8(ZERO);
  RAJA::ReduceMax<REDUCEPOLICY, double> cmax9(ZERO);
  RAJA::View<double, RAJA::Layout<4> > scview(network_data->scale, udata.nzl,
                                              udata.nyl, udata.nxl, udata.nchem);
  RAJA::View<double, RAJA::Layout<4> > wview(wdata, udata.nzl, udata.nyl, udata.nxl, udata.nchem);
  RAJA::kernel<XYZ_KERNEL_POL>(RAJA::make_tuple(RAJA::RangeSegment(0, udata.nzl),
                                                RAJA::RangeSegment(0, udata.nyl),
                                                RAJA::RangeSegment(0, udata.nxl)),
                               [=] RAJA_DEVICE (int k, int j, int i) {
    double tmp;
    tmp = scview(k,j,i,0)*wview(k,j,i,0);  cmean0 += tmp;  cmax0.max(tmp);  cmin0.min(tmp);
    tmp = scview(k,j,i,1)*wview(k,j,i,1);  cmean1 += tmp;  cmax1.max(tmp);  cmin1.min(tmp);
    tmp = scview(k,j,i,2)*wview(k,j,i,2);  cmean2 += tmp;  cmax2.max(tmp);  cmin2.min(tmp);
    tmp = scview(k,j,i,3)*wview(k,j,i,3);  cmean3 += tmp;  cmax3.max(tmp);  cmin3.min(tmp);
    tmp = scview(k,j,i,4)*wview(k,j,i,4);  cmean4 += tmp;  cmax4.max(tmp);  cmin4.min(tmp);
    tmp = scview(k,j,i,5)*wview(k,j,i,5);  cmean5 += tmp;  cmax5.max(tmp);  cmin5.min(tmp);
    tmp = scview(k,j,i,6)*wview(k,j,i,6);  cmean6 += tmp;  cmax6.max(tmp);  cmin6.min(tmp);
    tmp = scview(k,j,i,7)*wview(k,j,i,7);  cmean7 += tmp;  cmax7.max(tmp);  cmin7.min(tmp);
    tmp = scview(k,j,i,8)*wview(k,j,i,8);  cmean8 += tmp;  cmax8.max(tmp);  cmin8.min(tmp);
    tmp = scview(k,j,i,9)*wview(k,j,i,9);  cmean9 += tmp;  cmax9.max(tmp);  cmin9.min(tmp);
  });

  // print solutions at first location
  long int Ntot = udata.nxl * udata.nyl * udata.nzl;
  printf("  component:  H2I     H2II    HI      HII     HM      HeI     HeII    HeIII   de      ge\n");
  printf("        min: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         cmin0.get(), cmin1.get(), cmin2.get(), cmin3.get(), cmin4.get(),
         cmin5.get(), cmin6.get(), cmin7.get(), cmin8.get(), cmin9.get());
  printf("       mean: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         cmean0.get()/Ntot, cmean1.get()/Ntot, cmean2.get()/Ntot, cmean3.get()/Ntot,
         cmean4.get()/Ntot, cmean5.get()/Ntot, cmean6.get()/Ntot, cmean7.get()/Ntot,
         cmean8.get()/Ntot, cmean9.get()/Ntot);
  printf("        max: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         cmax0.get(), cmax1.get(), cmax2.get(), cmax3.get(), cmax4.get(),
         cmax5.get(), cmax6.get(), cmax7.get(), cmax8.get(), cmax9.get());
#else
  double *sc = network_data->scale[0];
  double cmean0, cmean1, cmean2, cmean3, cmean4, cmean5, cmean6, cmean7, cmean8, cmean9;
  cmean0 = cmean1 = cmean2 = cmean3 = cmean4 = cmean5 = cmean6 = cmean7 = cmean8 = cmean9 = ZERO;
  double cmax0, cmax1, cmax2, cmax3, cmax4, cmax5, cmax6, cmax7, cmax8, cmax9;
  cmax0 = cmax1 = cmax2 = cmax3 = cmax4 = cmax5 = cmax6 = cmax7 = cmax8 = cmax9 = ZERO;
  double cmin0, cmin1, cmin2, cmin3, cmin4, cmin5, cmin6, cmin7, cmin8, cmin9;
  cmin0 = cmin1 = cmin2 = cmin3 = cmin4 = cmin5 = cmin6 = cmin7 = cmin8 = cmin9 = 1e300;
  for (int k=0; k<udata.nzl; k++)
    for (int j=0; j<udata.nyl; j++)
      for (int i=0; i<udata.nxl; i++) {
        long int idx;
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean0 += sc[idx]*wdata[idx];
        cmax0  = max(cmax0, sc[idx]*wdata[idx]);
        cmin0  = min(cmin0, sc[idx]*wdata[idx]);

        idx = BUFIDX(1,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean1 += sc[idx]*wdata[idx];
        cmax1  = max(cmax1, sc[idx]*wdata[idx]);
        cmin1  = min(cmin1, sc[idx]*wdata[idx]);

        idx = BUFIDX(2,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean2 += sc[idx]*wdata[idx];
        cmax2  = max(cmax2, sc[idx]*wdata[idx]);
        cmin2  = min(cmin2, sc[idx]*wdata[idx]);

        idx = BUFIDX(3,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean3 += sc[idx]*wdata[idx];
        cmax3  = max(cmax3, sc[idx]*wdata[idx]);
        cmin3  = min(cmin3, sc[idx]*wdata[idx]);

        idx = BUFIDX(4,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean4 += sc[idx]*wdata[idx];
        cmax4  = max(cmax4, sc[idx]*wdata[idx]);
        cmin4  = min(cmin4, sc[idx]*wdata[idx]);

        idx = BUFIDX(5,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean5 += sc[idx]*wdata[idx];
        cmax5  = max(cmax5, sc[idx]*wdata[idx]);
        cmin5  = min(cmin5, sc[idx]*wdata[idx]);

        idx = BUFIDX(6,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean6 += sc[idx]*wdata[idx];
        cmax6  = max(cmax6, sc[idx]*wdata[idx]);
        cmin6  = min(cmin6, sc[idx]*wdata[idx]);

        idx = BUFIDX(7,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean7 += sc[idx]*wdata[idx];
        cmax7  = max(cmax7, sc[idx]*wdata[idx]);
        cmin7  = min(cmin7, sc[idx]*wdata[idx]);

        idx = BUFIDX(8,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean8 += sc[idx]*wdata[idx];
        cmax8  = max(cmax8, sc[idx]*wdata[idx]);
        cmin8  = min(cmin8, sc[idx]*wdata[idx]);

        idx = BUFIDX(9,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        cmean9 += sc[idx]*wdata[idx];
        cmax9  = max(cmax9, sc[idx]*wdata[idx]);
        cmin9  = min(cmin9, sc[idx]*wdata[idx]);
      }
  cmean0 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean1 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean2 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean3 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean4 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean5 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean6 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean7 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean8 /= (udata.nxl * udata.nyl * udata.nzl);
  cmean9 /= (udata.nxl * udata.nyl * udata.nzl);

  // print solutions at first location
  printf("  component:  H2I     H2II    HI      HII     HM      HeI     HeII    HeIII   de      ge\n");
  printf("        min: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         cmin0, cmin1, cmin2, cmin3, cmin4, cmin5, cmin6, cmin7, cmin8, cmin9);
  printf("       mean: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         cmean0, cmean1, cmean2, cmean3, cmean4, cmean5, cmean6, cmean7, cmean8, cmean9);
  printf("        max: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         cmax0, cmax1, cmax2, cmax3, cmax4, cmax5, cmax6, cmax7, cmax8, cmax9);
#endif
}


// dummy functions required for compilation (required when using Euler solver)
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  return(0);
}

//---- end of file ----
