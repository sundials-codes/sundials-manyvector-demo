/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Implementation file for shared main routine
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>

#ifdef DEBUG
#include "fenv.h"
#endif

// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem parameters
  long int N, Ntot, i;
  int Nsubvecs;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int dense_order;               // dense output order of accuracy
  int idense;                    // flag denoting integration type (dense output vs tstop)
  int imex;                      // flag denoting class of method (0=implicit, 1=explicit, 2=IMEX)
  int fixedpt;                   // flag denoting use of fixed-point nonlinear solver
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use (disabled if negative)
  N_Vector w = NULL;             // empty vectors for storing overall solution
  N_Vector *wsubvecs;
  void *arkode_mem = NULL;       // empty ARKStep memory structure

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // start run timer
  double tstart = MPI_Wtime();

  // read problem and solver parameters from input file / command line
  UserData udata;
  ARKodeParameters opts;
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (retval > 0) MPI_Abort(MPI_COMM_WORLD, 0);
  realtype dTout = (udata.tf-udata.t0)/(udata.nout);
  realtype tinout = MPI_Wtime() - tstart;

  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);

  // Output problem setup information
  bool outproc = (udata.myid == 0);
  if (outproc) {
    cout << "\n3D compressible inviscid Euler test problem:\n";
    cout << "   nprocs: " << udata.nprocs << " (" << udata.npx << " x "
         << udata.npy << " x " << udata.npz << ")\n";
    cout << "   spatial domain: [" << udata.xl << ", " << udata.xr << "] x ["
         << udata.yl << ", " << udata.yr << "] x ["
         << udata.zl << ", " << udata.zr << "]\n";
    cout << "   time domain = (" << udata.t0 << ", " << udata.tf << "]\n";
    cout << "   bdry cond (" << BC_PERIODIC << "=per, " << BC_NEUMANN << "=Neu, "
         << BC_DIRICHLET << "=Dir, " << BC_REFLECTING << "=refl): ["
         << udata.xlbc << ", " << udata.xrbc << "] x ["
         << udata.ylbc << ", " << udata.yrbc << "] x ["
         << udata.zlbc << ", " << udata.zrbc << "]\n";
    cout << "   gamma: " << udata.gamma << "\n";
    cout << "   cfl fraction: " << udata.cfl << "\n";
    cout << "   tracers/chemical species: " << udata.nchem << "\n";
    cout << "   spatial grid: " << udata.nx << " x " << udata.ny << " x "
         << udata.nz << "\n";
    if (restart >= 0)
      cout << "   restarting from output number: " << restart << "\n";
  }
  if (udata.showstats) {
    retval = MPI_Barrier(udata.comm);
    if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
    printf("      proc %4i: %li x %li x %li\n", udata.myid, udata.nxl, udata.nyl, udata.nzl);
    retval = MPI_Barrier(udata.comm);
    if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  }

  // open solver diagnostics output file for writing
  FILE *DFID = NULL;
  if (outproc)
    DFID=fopen("diags_euler3D.txt","w");

  // Initialize N_Vector data structures
  N = (udata.nxl)*(udata.nyl)*(udata.nzl);
  Ntot = (udata.nx)*(udata.ny)*(udata.nz);
  Nsubvecs = 5 + ((udata.nchem > 0) ? 1 : 0);
  wsubvecs = new N_Vector[Nsubvecs];
  for (i=0; i<5; i++) {
    wsubvecs[i] = NULL;
    wsubvecs[i] = N_VNew_Parallel(udata.comm, N, Ntot);
    if (check_flag((void *) wsubvecs[i], "N_VNew_Parallel (main)", 0)) MPI_Abort(udata.comm, 1);
  }
  if (udata.nchem > 0) {
    wsubvecs[5] = NULL;
    wsubvecs[5] = N_VNew_Serial(N*udata.nchem);
    if (check_flag((void *) wsubvecs[5], "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
  }
  w = N_VNew_MPIManyVector(Nsubvecs, wsubvecs);  // combined solution vector
  if (check_flag((void *) w, "N_VNew_MPIManyVector (main)", 0)) MPI_Abort(udata.comm, 1);

  // set initial conditions (or restart from file -- not yet operational)
  if (restart < 0) {
    retval = initial_conditions(udata.t0, w, udata);
    if (check_flag(&retval, "initial_conditions (main)", 1)) MPI_Abort(udata.comm, 1);
    restart = 0;
  } else {
    retval = read_restart(restart, udata.t0, w, udata);
    if (check_flag(&retval, "read_restart (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // initialize the integrator memory
  arkode_mem = ARKStepCreate(fEuler, NULL, udata.t0, w);
  if (check_flag((void*) arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);

  // setup the ARKStep integrator based on inputs

  //    pass udata to user functions
  retval = ARKStepSetUserData(arkode_mem, (void *) (&udata));
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
  retval = ARKStepSStolerances(arkode_mem, opts.rtol, opts.atol);
  if (check_flag(&retval, "ARKStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);

  //    supply cfl-stable step routine (if requested)
  if (udata.cfl > ZERO) {
    retval = ARKStepSetStabilityFn(arkode_mem, stability, (void *) (&udata));
    if (check_flag(&retval, "ARKStepSetStabilityFn (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // Initial batch of outputs
  double iostart = MPI_Wtime();
  //    Output initial conditions to disk
  retval = output_solution(udata.t0, w, opts.h0, restart, udata, opts);
  if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);

  //    Optionally output total mass/energy
  if (udata.showstats) {
    retval = check_conservation(udata.t0, w, udata);
    if (check_flag(&retval, "check_conservation (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    Output problem-specific diagnostic information
  retval = output_diagnostics(udata.t0, w, udata);
  if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);
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
  if (udata.showstats) {
    retval = print_stats(t, w, 0, arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
  }
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
    if (udata.showstats) {
      retval = print_stats(t, w, 1, arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    // output diagnostic information (if applicable)
    retval = output_diagnostics(t, w, udata);
    if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

    // output results to disk -- get current step from ARKStep first
    retval = ARKStepGetCurrentStep(arkode_mem, &hcur);
    if (check_flag(&retval, "ARKStepGetCurrentStep (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = output_solution(t, w, hcur, iout+1, udata, opts);
    if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
    tinout += MPI_Wtime() - iostart;

  }
  if (udata.showstats) {
    iostart = MPI_Wtime();
    retval = print_stats(t, w, 2, arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    tinout += MPI_Wtime() - iostart;
  }
  if (outproc) fclose(DFID);

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
  N_VDestroy(w);               // Free solution vectors
  for (i=0; i<Nsubvecs; i++)
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  ARKStepFree(&arkode_mem);    // Free integrator memory
  MPI_Finalize();              // Finalize MPI
  return 0;
}

//---- end of file ----
