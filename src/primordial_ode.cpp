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

#ifdef DEBUG
#include "fenv.h"
#endif

// utility function prototypes
void print_info(realtype &t, N_Vector w, final__data *network_data, EulerData &udata);


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
  final__data *network_data = final__setup_data(NULL, NULL);
  //    overwrite internal strip size
  network_data->nstrip = nstrip;
  //    set redshift value for non-cosmological run
  network_data->current_z = -1.0;

  // initialize N_Vector data structures
  N = (udata.nchem)*nstrip;
  w = atols = NULL;
  w = N_VNew_Serial(N);
  if (check_flag((void *) w, "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
  atols = N_VNew_Serial(N);
  if (check_flag((void *) atols, "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);

  // set initial conditions -- essentially-neutral primordial gas
  realtype density = 1e7;
  realtype tiny = 1e-10;
  realtype Hfrac = 0.74;
  realtype HI_weight = 1.00794;
  realtype HII_weight = 1.00794;
  realtype HM_weight = 1.00794;
  realtype HeI_weight = 4.002602;
  realtype HeII_weight = 4.002602;
  realtype HeIII_weight = 4.002602;
  realtype H2I_weight = 2*HI_weight;
  realtype H2II_weight = 2*HI_weight;
  realtype gamma = 5.0/3.0;
  realtype tevk = 1.1605e4;
  realtype mh = 1.67e-24;
  realtype kboltz = 1.380e-16;
  realtype nH2I, nH2II, nHI, nHII, nHM, nHeI, nHeII, nHeIII, ndens;
  realtype H2I, H2II, HI, HII, HM, HeI, HeII, HeIII, de, T, ge;
  realtype *wdata = NULL;
  wdata = N_VGetArrayPointer(w);
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {

        // set desired number densities
        nH2I = tiny;
        nH2II = tiny;
        nHII = tiny;
        nHM = tiny;
        nHeII = tiny;
        nHeIII = tiny;

        // set mass densities into local variables
        H2I = nH2I*H2I_weight;
        H2II = nH2II*H2II_weight;
        HII = nHII*HII_weight;
        HM = nHM*HM_weight;
        HeII = nHeII*HeII_weight;
        HeIII = nHeIII*HeIII_weight;
        HeI = (ONE-Hfrac)*density - HeII - HeIII;
        HI = density - (H2I+H2II+HII+HM+HeI+HeII+HeIII);

        // compute derived number densities
        nHeI = HeI/HeI_weight;
        nHI = HI/HI_weight;
        ndens = nH2I + nH2II + nHII + nHM + nHeII + nHeIII + nHeI + nHI;
        de = nHII + nHeII + 2*nHeIII - nHM + nH2II;

        // set varying temperature throughout domain, and convert to gas energy
        T = 1e3 + 10.0*(i+udata.is-udata.nx/2)/(udata.nx-1)
                + 10.0*(j+udata.js-udata.ny/2)/(udata.ny-1)
                + 10.0*(k+udata.ks-udata.nz/2)/(udata.nz-1);
        ge = (T * ndens * kboltz) / (density * mh * (gamma - ONE));

        // copy final results into vector: H2_1, H2_2, H_1, H_2, H_m0, He_1, He_2, He_3, de, ge
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        wdata[idx+0] = H2I;
        wdata[idx+1] = H2II;
        wdata[idx+2] = HI;
        wdata[idx+3] = HII;
        wdata[idx+4] = HM;
        wdata[idx+5] = HeI;
        wdata[idx+6] = HeII;
        wdata[idx+7] = HeIII;
        wdata[idx+8] = de;
        wdata[idx+9] = ge;

      }

  // set absolute tolerance array
  realtype *atdata = NULL;
  atdata = N_VGetArrayPointer(atols);
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
        atdata[idx+0] = opts.atol * H2I_weight;   // H2I
        atdata[idx+0] = opts.atol * H2II_weight;  // H2II
        atdata[idx+0] = opts.atol * HI_weight;    // HI
        atdata[idx+0] = opts.atol * HII_weight;   // HII
        atdata[idx+0] = opts.atol * HM_weight;    // HM
        atdata[idx+0] = opts.atol * HeI_weight;   // HeI
        atdata[idx+0] = opts.atol * HeII_weight;  // HeII
        atdata[idx+0] = opts.atol * HeIII_weight; // HeIII
        atdata[idx+0] = opts.atol;                // de
        atdata[idx+0] = opts.atol;                // ge
      }

  // store mass densities in network_data structure, and rescale initial conditions
  for (k=0, idx=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++,idx++) {
        network_data->mdensity[idx] = ZERO;
        idx2 = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);

        // H2I
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= H2I_weight;

        // H2II
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= H2II_weight;

        // HI
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= HI_weight;

        // HII
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= HII_weight;

        // HM
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= HM_weight;

        // HeI
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= HeI_weight;

        // HeII
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= HeII_weight;

        // HeIII
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= HeIII_weight;

        // de
        network_data->mdensity[idx] += wdata[idx2];
        wdata[idx2++] /= ONE;

        // final values
        network_data->mdensity[idx] *= mh;
        network_data->inv_mdensity[idx]  = ONE / network_data->mdensity[idx];
      }

  // move input solution values into 'scale' components of network_data structure
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++)
        for (l=0; l<udata.nchem; l++) {
          idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
          network_data->scale[idx] = wdata[idx];
          network_data->inv_scale[idx] = ONE / wdata[idx];
          wdata[idx] = ONE;
        }


  // initialize the integrator memory
  arkode_mem = ARKStepCreate(NULL, calculate_rhs_final_, udata.t0, w);
  if (check_flag((void*) arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);

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
  retval = ARKStepSetLinearSolver(arkode_mem, LS, A);
  if (check_flag(&retval, "ARKStepSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepSetJacFn(arkode_mem, calculate_sparse_jacobian_final_);
  if (check_flag(&retval, "ARKStepSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);

  // setup the ARKStep integrator based on inputs

  //    pass network_udata to user functions
  retval = ARKStepSetUserData(arkode_mem, network_data);
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  //    set diagnostics file
  if (outproc) {
    retval = ARKStepSetDiagnostics(arkode_mem, DFID);
    if (check_flag(&retval, "ARKStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    set RK order, or specify individual Butcher table -- "order" overrides "btable"
  if (opts.order != 0) {
    retval = ARKStepSetOrder(arkode_mem, opts.order);
    if (check_flag(&retval, "ARKStepSetOrder (main)", 1)) MPI_Abort(udata.comm, 1);
  } else if (opts.btable != -1) {
    retval = ARKStepSetTableNum(arkode_mem, -1, opts.btable);
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


  // Initial batch of outputs
  double iostart = MPI_Wtime();
  //    Output initial conditions to disk (IMPLEMENT LATER, IF DESIRED)

  //    Output problem-specific diagnostic information
  print_info(udata.t0, w, network_data, udata);
  tinout += MPI_Wtime() - iostart;

  // If (dense_order == -1), use tstop mode
  if (opts.dense_order == -1)
    idense = 0;
  else   // otherwise tell integrator to use dense output
    idense = 1;

  // compute overall setup time
  double tsetup = MPI_Wtime() - tstart;
  tstart = MPI_Wtime();

  /* Main time-stepping loop: calls ARKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  realtype t = udata.t0;
  realtype tout = udata.t0+dTout;
  realtype hcur;
  int iout;
  for (iout=restart; iout<restart+udata.nout; iout++) {

    if (!idense)
      retval = ARKStepSetStopTime(arkode_mem, tout);
    retval = ARKStepEvolve(arkode_mem, tout, w, &t, ARK_NORMAL);  // call integrator
    if (retval >= 0) {                                            // successful solve: update output time
      tout = min(tout+dTout, udata.tf);
    } else {                                                      // unsuccessful solve: break
      if (outproc)
	cerr << "Solver failure, stopping integration\n";
      return 1;
    }

    iostart = MPI_Wtime();
    // output statistics to stdout
    print_info(t, w, network_data, udata);

    // output results to disk -- get current step from ARKStep first
    retval = ARKStepGetCurrentStep(arkode_mem, &hcur);
    if (check_flag(&retval, "ARKStepGetCurrentStep (main)", 1)) MPI_Abort(udata.comm, 1);
    // TODO
    tinout += MPI_Wtime() - iostart;

  }
  if (outproc) fclose(DFID);


  // reconstruct overall solution values
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        idx = BUFIDX(0,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);

        // H2I
        wdata[idx] *= (network_data->scale[idx])*H2I_weight;
        idx++;

        // H2II
        wdata[idx] *= (network_data->scale[idx])*H2II_weight;
        idx++;

        // HI
        wdata[idx] *= (network_data->scale[idx])*HI_weight;
        idx++;

        // HII
        wdata[idx] *= (network_data->scale[idx])*HII_weight;
        idx++;

        // HM
        wdata[idx] *= (network_data->scale[idx])*HM_weight;
        idx++;

        // HeI
        wdata[idx] *= (network_data->scale[idx])*HeI_weight;
        idx++;

        // HeII
        wdata[idx] *= (network_data->scale[idx])*HeII_weight;
        idx++;

        // HeIII
        wdata[idx] *= (network_data->scale[idx])*HeIII_weight;
        idx++;

        // de
        wdata[idx] *= (network_data->scale[idx]);
        idx++;

        // ge
        wdata[idx] *= (network_data->scale[idx]);
      }

  // re-calculate temperature at the final output
  final__calculate_temperature(network_data, wdata, nstrip, udata.nchem);

  // compute simulation time
  double tsimul = MPI_Wtime() - tstart;

  // Print some final statistics
  long int nst, nst_a, nfe, nfi, netf;
  nst = nst_a = nfe = nfi = netf = 0;
  retval = ARKStepGetNumSteps(arkode_mem, &nst);
  if (check_flag(&retval, "ARKStepGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumStepAttempts(arkode_mem, &nst_a);
  if (check_flag(&retval, "ARKStepGetNumStepAttempts (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumRhsEvals(arkode_mem, &nfe, &nfi);
  if (check_flag(&retval, "ARKStepGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(&retval, "ARKStepGetNumErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);

  if (outproc) {
    cout << "\nFinal Solver Statistics:\n";
    cout << "   Internal solver steps = " << nst << " (attempted = " << nst_a << ")\n";
    cout << "   Total RHS evals:  Fe = " << nfe << ",  Fi = " << nfi << "\n";
    cout << "   Total number of error test failures = " << netf << "\n";
    cout << "   Total setup time = " << tsetup << "\n";
    cout << "   Total I/O time = " << tinout << "\n";
    cout << "   Total simulation time = " << tsimul << "\n";
  }

  // Output mass/energy conservation error
  if (udata.showstats) {
    retval = check_conservation(t, w, udata);
    if (check_flag(&retval, "check_conservation (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // Clean up and return with successful completion
  N_VDestroy(w);               // Free solution and absolute tolerance vectors
  N_VDestroy(atols);
  SUNLinSolFree(LS);           // Free matrix and linear solver
  SUNMatDestroy(A);
  ARKStepFree(&arkode_mem);    // Free integrator memory
  MPI_Finalize();              // Finalize MPI
  return 0;
}


void print_info(realtype &t, N_Vector w, final__data *network_data, EulerData &udata)
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
  realtype HI_weight = 1.00794;
  realtype HII_weight = 1.00794;
  realtype HM_weight = 1.00794;
  realtype HeI_weight = 4.002602;
  realtype HeII_weight = 4.002602;
  realtype HeIII_weight = 4.002602;
  realtype H2I_weight = 2*HI_weight;
  realtype H2II_weight = 2*HI_weight;

  // print solutions at first location
  printf("\nt = %.3e\n", t);
  printf("  T[%li,%li,%li] = %.5e\n", i1, j1, k1, network_data->Ts[idx1]);
  printf("  species: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         network_data->scale[idx1+0]*wdata[idx1+0]*H2I_weight,
         network_data->scale[idx1+1]*wdata[idx1+1]*H2II_weight,
         network_data->scale[idx1+2]*wdata[idx1+2]*HI_weight,
         network_data->scale[idx1+3]*wdata[idx1+3]*HII_weight,
         network_data->scale[idx1+4]*wdata[idx1+4]*HM_weight,
         network_data->scale[idx1+5]*wdata[idx1+5]*HeI_weight,
         network_data->scale[idx1+6]*wdata[idx1+6]*HeII_weight,
         network_data->scale[idx1+7]*wdata[idx1+7]*HeIII_weight,
         network_data->scale[idx1+8]*wdata[idx1+8],
         network_data->scale[idx1+9]*wdata[idx1+9]);

  // print solutions at second location
  printf("  T[%li,%li,%li] = %.5e\n", i2, j2, k2, network_data->Ts[idx2]);
  printf("  species: %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e\n",
         network_data->scale[idx2+0]*wdata[idx2+0]*H2I_weight,
         network_data->scale[idx2+1]*wdata[idx2+1]*H2II_weight,
         network_data->scale[idx2+2]*wdata[idx2+2]*HI_weight,
         network_data->scale[idx2+3]*wdata[idx2+3]*HII_weight,
         network_data->scale[idx2+4]*wdata[idx2+4]*HM_weight,
         network_data->scale[idx2+5]*wdata[idx2+5]*HeI_weight,
         network_data->scale[idx2+6]*wdata[idx2+6]*HeII_weight,
         network_data->scale[idx2+7]*wdata[idx2+7]*HeIII_weight,
         network_data->scale[idx2+8]*wdata[idx2+8],
         network_data->scale[idx2+9]*wdata[idx2+9]);
}


// dummy functions required for compilation (required when using Euler solver)
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  return(0);
}

//---- end of file ----
