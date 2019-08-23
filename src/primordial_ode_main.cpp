/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Implementation file to test Dengo interface -- note that although
 this uses the EulerData structure, we do not actually create the
 fluid fields all
 spatial domain and MPI information is ignored.  We only use this
 infrastructure to enable simplified input of the time interval
 and ARKode solver options.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <dengo_primordial_network.hpp>
#ifdef CVKLU
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#else
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#endif

#ifdef USE_CVODE
#include <cvode/cvode.h>
#include <cvode/cvode_ls.h>
#else
#include <arkode/arkode_arkstep.h>
#endif


#ifdef DEBUG
#include "fenv.h"
#endif

// utility function prototypes
void print_info(void *arkode_mem, realtype &t, N_Vector w,
                cvklu_data *network_data, EulerData &udata);


// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem parameters
  long int N, i, j, k, l, idx, idx2, nstrip;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int dense_order;               // dense output order of accuracy
  int idense;                    // flag denoting integration type (dense output vs tstop)
  int imex;                      // flag denoting class of method (0=implicit, 1=explicit, 2=IMEX)
  int fixedpt;                   // flag denoting use of fixed-point nonlinear solver
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use (disabled here)
  int nprocs;                    // total number of MPI processes
  N_Vector w = NULL;             // empty vectors for storing overall solution, absolute tolerance array
  N_Vector atols = NULL;
  SUNLinearSolver LS = NULL;     // empty linear solver and matrix structures
  SUNMatrix A = NULL;
  void *arkode_mem = NULL;       // empty ARKStep memory structure

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

  // start run timer
  double tstart = MPI_Wtime();

  // read problem and solver parameters from input file / command line
  EulerData udata;
  ARKodeParameters opts;
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (retval > 0) MPI_Abort(MPI_COMM_WORLD, 0);
  realtype dTout = (udata.tf-udata.t0)/(udata.nout);
  realtype tinout = MPI_Wtime() - tstart;

  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);
  udata.nchem = 10;   // primordial network requires 10 fields:
                      //   H2_1, H2_2, H_1, H_2, H_m0, He_1, He_2, He_3, de, ge

  // set nstrip value to all cells on this process
  nstrip = udata.nxl * udata.nyl * udata.nzl;

  // ensure that nxl*nyl*nzl (inputs) <= MAX_NCELLS (dengo preprocessor value)
  if (nstrip > MAX_NCELLS) {
    cerr << "primordial_ode error: total spatial subdomain size (" <<
      nstrip << ") exceeds maximum (" << MAX_NCELLS << ")\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

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
  }

  // open solver diagnostics output file for writing
  FILE *DFID = NULL;
  if (outproc)
    DFID=fopen("diags_primordial_ode.txt","w");

  // initialize primordial rate tables, etc
  cvklu_data *network_data = cvklu_setup_data("primordial_tables.h5", NULL, NULL);
  //    overwrite internal strip size
  network_data->nstrip = nstrip;
  //    set redshift value for non-cosmological run
  network_data->current_z = -1.0;

  // initialize N_Vector data structures
  N = (udata.nchem)*nstrip;
  w = N_VNew_Serial(N);
  if (check_flag((void *) w, "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
  atols = N_VNew_Serial(N);
  if (check_flag((void *) atols, "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);

  // set initial conditions -- essentially-neutral primordial gas
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
  realtype *wdata = NULL;
  wdata = N_VGetArrayPointer(w);
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

        // copy final results into vector: H2_1, H2_2, H_1, H_2, H_m0, He_1, He_2, He_3, de, ge;
        // converting to 'dimensionless' electron number density
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        wdata[idx+0] = nH2I;
        wdata[idx+1] = nH2II;
        wdata[idx+2] = nHI;
        wdata[idx+3] = nHII;
        wdata[idx+4] = nHM;
        wdata[idx+5] = nHeI;
        wdata[idx+6] = nHeII;
        wdata[idx+7] = nHeIII;
        wdata[idx+8] = de / m_amu;
        wdata[idx+9] = ge;

      }

  // set absolute tolerance array
  realtype *atdata = NULL;
  atdata = N_VGetArrayPointer(atols);
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        atdata[idx+0] = opts.atol; // H2I
        atdata[idx+1] = opts.atol; // H2II
        atdata[idx+2] = opts.atol; // HI
        atdata[idx+3] = opts.atol; // HII
        atdata[idx+4] = opts.atol; // HM
        atdata[idx+5] = opts.atol; // HeI
        atdata[idx+6] = opts.atol; // HeII
        atdata[idx+7] = opts.atol; // HeIII
        atdata[idx+8] = opts.atol; // de
        atdata[idx+9] = opts.atol; // ge
      }

  // move input solution values into 'scale' components of network_data structure
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++)
        for (l=0; l<udata.nchem; l++) {
          idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
          network_data->scale[0][idx] = wdata[idx];
          network_data->inv_scale[0][idx] = ONE / wdata[idx];
          wdata[idx] = ONE;
        }

  // compute auxiliary values within network_data structure
  setting_up_extra_variables(network_data, network_data->scale[0], nstrip);

  // initialize the integrator memory
#ifdef USE_CVODE
  arkode_mem = CVodeCreate(CV_BDF);
  if (check_flag((void*) arkode_mem, "CVodeCreate (main)", 0)) MPI_Abort(udata.comm, 1);
  retval = CVodeInit(arkode_mem, calculate_rhs_cvklu, udata.t0, w);
  if (check_flag(&retval, "CVodeInit (main)", 1)) MPI_Abort(udata.comm, 1);
#else
  arkode_mem = ARKStepCreate(NULL, calculate_rhs_cvklu, udata.t0, w);
  if (check_flag((void*) arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);
#endif

  // create matrix and linear solver modules, and attach to ARKStep
#ifdef CVKLU
  A  = SUNSparseMatrix(N, N, 64*nstrip, CSR_MAT);
  if (check_flag((void*) A, "SUNSparseMatrix (main)", 0)) MPI_Abort(udata.comm, 1);
  LS = SUNLinSol_KLU(w, A);
  if (check_flag((void*) LS, "SUNLinSol_KLU (main)", 0)) MPI_Abort(udata.comm, 1);
#else
  A  = SUNDenseMatrix(N, N);
  if (check_flag((void*) A, "SUNDenseMatrix (main)", 0)) MPI_Abort(udata.comm, 1);
  LS = SUNLinSol_Dense(w, A);
  if (check_flag((void*) LS, "SUNLinSol_Dense (main)", 0)) MPI_Abort(udata.comm, 1);
#endif
#ifdef USE_CVODE
  retval = CVodeSetLinearSolver(arkode_mem, LS, A);
  if (check_flag(&retval, "CVodeSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeSetJacFn(arkode_mem, calculate_sparse_jacobian_cvklu);
  if (check_flag(&retval, "CVodeSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);
#else
  retval = ARKStepSetLinearSolver(arkode_mem, LS, A);
  if (check_flag(&retval, "ARKStepSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepSetJacFn(arkode_mem, calculate_sparse_jacobian_cvklu);
  if (check_flag(&retval, "ARKStepSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);
#endif

  // setup the ARKStep integrator based on inputs

#ifdef USE_CVODE
  //    pass network_udata to user functions
  retval = CVodeSetUserData(arkode_mem, network_data);
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
  retval = ARKStepSetUserData(arkode_mem, network_data);
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set diagnostics file
  if (outproc) {
    retval = ARKStepSetDiagnostics(arkode_mem, DFID);
    if (check_flag(&retval, "ARKStepSetDiagnostics (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    set RK order, or specify individual Butcher table -- "order" overrides "btable"
  if (opts.order != 0) {
    retval = ARKStepSetOrder(arkode_mem, opts.order);
    if (check_flag(&retval, "ARKStepSetOrder (main)", 1)) MPI_Abort(udata.comm, 1);
  } else if (opts.btable != -1) {
    retval = ARKStepSetTableNum(arkode_mem, opts.btable, -1);
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
  double iostart = MPI_Wtime();
  //    Output initial conditions to disk (IMPLEMENT LATER, IF DESIRED)

  //    Output problem-specific diagnostic information
  print_info(arkode_mem, udata.t0, w, network_data, udata);
  tinout += MPI_Wtime() - iostart;

  // If (dense_order == -1), use tstop mode
  if (opts.dense_order == -1)
    idense = 0;
  else   // otherwise tell integrator to use dense output
    idense = 1;

  // compute overall setup time
  double tsetup = MPI_Wtime() - tstart;
  tstart = MPI_Wtime();

  N_Vector ele     = N_VNew_Serial(N);
  N_Vector eweight = N_VNew_Serial(N);
  N_Vector etest   = N_VNew_Serial(N);

  /* Main time-stepping loop: calls ARKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  realtype t = udata.t0;
  realtype tout = udata.t0+dTout;
  int iout;
  for (iout=restart; iout<restart+udata.nout; iout++) {

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

    iostart = MPI_Wtime();
    // output statistics to stdout
    print_info(arkode_mem, t, w, network_data, udata);
#ifdef USE_CVODE
    retval = CVodeGetEstLocalErrors(arkode_mem, ele);
    retval = CVodeGetErrWeights(arkode_mem, eweight);
#else
    retval = ARKStepGetEstLocalErrors(arkode_mem, ele);
    retval = ARKStepGetErrWeights(arkode_mem, eweight);
#endif
    N_VProd(ele, eweight, etest);
    N_VAbs(etest, etest);
    if (N_VMaxNorm(etest) > 0.2) {
      cout << "  Main components of local error:\n";
      for (k=0; k<udata.nzl; k++)
        for (j=0; j<udata.nyl; j++)
          for (i=0; i<udata.nxl; i++)
            for (l=0; l<udata.nchem; l++) {
              idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
              if (NV_Ith_S(etest, idx) > 0.2) {
                cout << "    etest(" << i << "," << j << "," << k << "," << l << ") = "
                     << NV_Ith_S(etest, idx) << "\n";
              }
            }
      cout << "\n";
    }

    // output results to disk
    // TODO
    tinout += MPI_Wtime() - iostart;

  }
  if (outproc) fclose(DFID);


  // reconstruct overall solution values, converting back to mass densities
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);

        // H2I
        wdata[idx] *= (network_data->scale[0][idx]) * H2I_weight;
        idx++;

        // H2II
        wdata[idx] *= (network_data->scale[0][idx]) * H2II_weight;
        idx++;

        // HI
        wdata[idx] *= (network_data->scale[0][idx]) * HI_weight;
        idx++;

        // HII
        wdata[idx] *= (network_data->scale[0][idx]) * HII_weight;
        idx++;

        // HM
        wdata[idx] *= (network_data->scale[0][idx]) * HM_weight;
        idx++;

        // HeI
        wdata[idx] *= (network_data->scale[0][idx]) * HeI_weight;
        idx++;

        // HeII
        wdata[idx] *= (network_data->scale[0][idx]) * HeII_weight;
        idx++;

        // HeIII
        wdata[idx] *= (network_data->scale[0][idx]) * HeIII_weight;
        idx++;

        // de
        wdata[idx] *= (network_data->scale[0][idx]) * m_amu;
        idx++;

        // ge
        wdata[idx] *= (network_data->scale[0][idx]);
      }

  // compute simulation time
  double tsimul = MPI_Wtime() - tstart;

  // Print some final statistics
  long int nst, nst_a, nfe, nfi, netf, nls, nni, ncf;
  nst = nst_a = nfe = nfi = netf = nls = nni = ncf = 0;
#ifdef USE_CVODE
  retval = CVodeGetNumSteps(arkode_mem, &nst);
  if (check_flag(&retval, "CVodeGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumRhsEvals(arkode_mem, &nfi);
  if (check_flag(&retval, "CVodeGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(&retval, "CVodeGetNumErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumLinSolvSetups(arkode_mem, &nls);
  if (check_flag(&retval, "CVodeGetNumLinSolvSetups (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumNonlinSolvIters(arkode_mem, &nni);
  if (check_flag(&retval, "CVodeGetNumNonlinSolvIters (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = CVodeGetNumNonlinSolvConvFails(arkode_mem, &ncf);
  if (check_flag(&retval, "CVodeGetNumNonlinSolvConvFails (main)", 1)) MPI_Abort(udata.comm, 1);
#else
  retval = ARKStepGetNumSteps(arkode_mem, &nst);
  if (check_flag(&retval, "ARKStepGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumStepAttempts(arkode_mem, &nst_a);
  if (check_flag(&retval, "ARKStepGetNumStepAttempts (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumRhsEvals(arkode_mem, &nfe, &nfi);
  if (check_flag(&retval, "ARKStepGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(&retval, "ARKStepGetNumErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinSolvSetups(arkode_mem, &nls);
  if (check_flag(&retval, "ARKStepGetNumLinSolvSetups (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNonlinSolvStats(arkode_mem, &nni, &ncf);
  if (check_flag(&retval, "ARKStepGetNonlinSolvStats (main)", 1)) MPI_Abort(udata.comm, 1);
#endif

  if (outproc) {
    cout << "\nFinal Solver Statistics:\n";
    cout << "   Internal solver steps = " << nst << " (attempted = " << nst_a << ")\n";
    cout << "   Total RHS evals:  Fe = " << nfe << ",  Fi = " << nfi << "\n";
    cout << "   Total number of error test failures = " << netf << "\n";
    if (nls > 0)
      cout << "   Total number of lin solv setups = " << nls << "\n";
    if (nni > 0) {
      cout << "   Total number of nonlin iters = " << nni << "\n";
      cout << "   Total number of nonlin conv fails = " << ncf << "\n";
    }
    cout << "   Total setup time = " << tsetup << "\n";
    cout << "   Total I/O time = " << tinout << "\n";
    cout << "   Total simulation time = " << tsimul << "\n";
  }

  // Clean up and return with successful completion
  N_VDestroy(w);               // Free solution and absolute tolerance vectors
  N_VDestroy(atols);
  SUNLinSolFree(LS);           // Free matrix and linear solver
  SUNMatDestroy(A);
#ifdef USE_CVODE
  CVodeFree(&arkode_mem);      // Free integrator memory
#else
  ARKStepFree(&arkode_mem);    // Free integrator memory
#endif
  MPI_Finalize();              // Finalize MPI
  return 0;
}


// Prints out solution values at two locations (converting from
// number densities to mass units)
void print_info(void *arkode_mem, realtype &t, N_Vector w,
                cvklu_data *network_data, EulerData &udata)
{
  // indices to print
  long int i1 = udata.nxl/3;
  long int j1 = udata.nyl/3;
  long int k1 = udata.nzl/3;
  long int idx1 = BUFIDX(0,i1,j1,k1,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
  long int i2 = 2*udata.nxl/3;
  long int j2 = 2*udata.nyl/3;
  long int k2 = 2*udata.nzl/3;
  long int idx2 = BUFIDX(0,i2,j2,k2,udata.nchem,udata.nxl,udata.nyl,udata.nzl);

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
  realtype m_amu = 1.66053904e-24;

  // get current number of time steps
  long int nst = 0;
#ifdef USE_CVODE
  CVodeGetNumSteps(arkode_mem, &nst);
#else
  ARKStepGetNumSteps(arkode_mem, &nst);
#endif

  // print current time and number of steps
  printf("\nt = %.3e  (nst = %li)\n", t, nst);

  // print solutions at first location
  printf("  w[%li,%li,%li]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         i1, j1, k1,
         network_data->scale[0][idx1+0]*wdata[idx1+0],
         network_data->scale[0][idx1+1]*wdata[idx1+1],
         network_data->scale[0][idx1+2]*wdata[idx1+2],
         network_data->scale[0][idx1+3]*wdata[idx1+3],
         network_data->scale[0][idx1+4]*wdata[idx1+4],
         network_data->scale[0][idx1+5]*wdata[idx1+5],
         network_data->scale[0][idx1+6]*wdata[idx1+6],
         network_data->scale[0][idx1+7]*wdata[idx1+7],
         network_data->scale[0][idx1+8]*wdata[idx1+8],
         network_data->scale[0][idx1+9]*wdata[idx1+9]);
         // network_data->scale[0][idx1+0]*wdata[idx1+0]*H2I_weight,
         // network_data->scale[0][idx1+1]*wdata[idx1+1]*H2II_weight,
         // network_data->scale[0][idx1+2]*wdata[idx1+2]*HI_weight,
         // network_data->scale[0][idx1+3]*wdata[idx1+3]*HII_weight,
         // network_data->scale[0][idx1+4]*wdata[idx1+4]*HM_weight,
         // network_data->scale[0][idx1+5]*wdata[idx1+5]*HeI_weight,
         // network_data->scale[0][idx1+6]*wdata[idx1+6]*HeII_weight,
         // network_data->scale[0][idx1+7]*wdata[idx1+7]*HeIII_weight,
         // network_data->scale[0][idx1+8]*wdata[idx1+8]*m_amu,
         // network_data->scale[0][idx1+9]*wdata[idx1+9]);

  // print solutions at second location
  printf("  w[%li,%li,%li]: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         i2, j2, k2,
         network_data->scale[0][idx2+0]*wdata[idx2+0],
         network_data->scale[0][idx2+1]*wdata[idx2+1],
         network_data->scale[0][idx2+2]*wdata[idx2+2],
         network_data->scale[0][idx2+3]*wdata[idx2+3],
         network_data->scale[0][idx2+4]*wdata[idx2+4],
         network_data->scale[0][idx2+5]*wdata[idx2+5],
         network_data->scale[0][idx2+6]*wdata[idx2+6],
         network_data->scale[0][idx2+7]*wdata[idx2+7],
         network_data->scale[0][idx2+8]*wdata[idx2+8],
         network_data->scale[0][idx2+9]*wdata[idx2+9]);
         // network_data->scale[0][idx2+0]*wdata[idx2+0]*H2I_weight,
         // network_data->scale[0][idx2+1]*wdata[idx2+1]*H2II_weight,
         // network_data->scale[0][idx2+2]*wdata[idx2+2]*HI_weight,
         // network_data->scale[0][idx2+3]*wdata[idx2+3]*HII_weight,
         // network_data->scale[0][idx2+4]*wdata[idx2+4]*HM_weight,
         // network_data->scale[0][idx2+5]*wdata[idx2+5]*HeI_weight,
         // network_data->scale[0][idx2+6]*wdata[idx2+6]*HeII_weight,
         // network_data->scale[0][idx2+7]*wdata[idx2+7]*HeIII_weight,
         // network_data->scale[0][idx2+8]*wdata[idx2+8]*m_amu,
         // network_data->scale[0][idx2+9]*wdata[idx2+9]);
}


// dummy functions required for compilation (required when using Euler solver)
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  return(0);
}

//---- end of file ----
