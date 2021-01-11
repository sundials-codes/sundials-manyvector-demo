/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Main Euler equation driver:

 This evolves the 3D compressible, inviscid Euler equations.

 The problem is evolved using ARKode's ARKStep time-stepping
 module for a temporally adaptive explicit Runge--Kutta solve.
 Nearly all adaptivity options are controllable via user inputs.
 If the input file specifies fixedstep=1, then temporal adaptivity
 is disabled, and the solver will use the fixed step size
 h=hmax.  In this case, if the input file specifies htrans>0,
 then temporal adaptivity will be used for the start of the
 simulation [t0,t0+htrans], followed by fixed time-stepping using
 h=hmax.  We require that htrans is smaller than the first
 output time interval, i.e., t0+htrans < t0+dTout.

 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <arkode/arkode_arkstep.h>

#ifdef DEBUG
#include "fenv.h"
#endif

// macros for handling formatting of diagnostic output (1 enable, 0 disable)
#define PRINT_CGS 1
#define PRINT_SCIENTIFIC 1


// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem variables
  long int N, Ntot, i;
  int Nsubvecs;
  int retval;                    // reusable error-checking flag
  int idense;                    // flag denoting integration type (dense output vs tstop)
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use (disabled if negative)
  N_Vector w = NULL;             // empty vectors for storing overall solution
  N_Vector *wsubvecs;
  void *arkode_mem = NULL;       // empty ARKStep memory structure
  EulerData udata;               // solver data structures
  ARKodeParameters opts;

  //--- General Initialization ---//

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // start various code profilers
  retval = udata.profile[PR_SETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  retval = udata.profile[PR_IO].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

   // read problem and solver parameters from input file / command line
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (retval > 0) MPI_Abort(MPI_COMM_WORLD, 0);
  realtype dTout = (udata.tf-udata.t0)/(udata.nout);
  retval = udata.profile[PR_IO].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // if fixed time stepping is specified, ensure that hmax>0
  if (opts.fixedstep && (opts.hmax <= ZERO)) {
    if (udata.myid == 0)
      cerr << "\nError: fixed time stepping requires hmax > 0 ("
           << opts.hmax << " given)\n";
    MPI_Abort(udata.comm, 1);
  }

  // update fixedstep parameter when initial transient evolution is requested
  if (opts.fixedstep && (opts.htrans>0))  opts.fixedstep=2;

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
    if (opts.fixedstep > 0)
      cout << "   fixed timestep size: " << opts.hmax << "\n";
    if (opts.fixedstep == 2)
      cout << "   initial transient evolution: " << opts.htrans << "\n";
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
    if (opts.fusedkernels) {
      cout << "   fused N_Vector kernels enabled\n";
    } else {
      cout << "   fused N_Vector kernels disabled\n";
    }
    if (opts.localreduce) {
      cout << "   local N_Vector reduction operations enabled\n";
    } else {
      cout << "   local N_Vector reduction operations disabled\n";
    }
    if (restart >= 0)
      cout << "   restarting from output number: " << restart << "\n";
  }
#ifdef DEBUG
  if (udata.showstats) {
    retval = MPI_Barrier(udata.comm);
    if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
    printf("      proc %4i: %li x %li x %li\n", udata.myid, udata.nxl, udata.nyl, udata.nzl);
    retval = MPI_Barrier(udata.comm);
    if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  }
#endif

  // open solver diagnostics output file for writing
  FILE *DFID = NULL;
  if (udata.showstats && outproc) {
    DFID=fopen("diags_euler3D.txt","w");
  }

  // Initialize N_Vector data structures with configured vector operations
  N = (udata.nxl)*(udata.nyl)*(udata.nzl);
  Ntot = (udata.nx)*(udata.ny)*(udata.nz);
  Nsubvecs = 5 + ((udata.nchem > 0) ? 1 : 0);
  wsubvecs = new N_Vector[Nsubvecs];
  for (i=0; i<5; i++) {
    wsubvecs[i] = NULL;
    wsubvecs[i] = N_VNew_Parallel(udata.comm, N, Ntot);
    if (check_flag((void *) wsubvecs[i], "N_VNew_Parallel (main)", 0)) MPI_Abort(udata.comm, 1);
    retval = N_VEnableFusedOps_Parallel(wsubvecs[i], opts.fusedkernels);
    if (check_flag(&retval, "N_VEnableFusedOps_Parallel (main)", 1)) MPI_Abort(udata.comm, 1);
    if (opts.localreduce == 0) {
      wsubvecs[i]->ops->nvdotprodlocal = NULL;
      wsubvecs[i]->ops->nvmaxnormlocal = NULL;
      wsubvecs[i]->ops->nvminlocal = NULL;
      wsubvecs[i]->ops->nvl1normlocal = NULL;
      wsubvecs[i]->ops->nvinvtestlocal = NULL;
      wsubvecs[i]->ops->nvconstrmasklocal = NULL;
      wsubvecs[i]->ops->nvminquotientlocal = NULL;
      wsubvecs[i]->ops->nvwsqrsumlocal = NULL;
      wsubvecs[i]->ops->nvwsqrsummasklocal = NULL;
    }
  }
  if (udata.nchem > 0) {
    wsubvecs[5] = NULL;
#ifdef USERAJA
    wsubvecs[5] = N_VNewManaged_Raja(N*udata.nchem);
    if (check_flag((void *) wsubvecs[5], "N_VNewManaged_Raja (main)", 0)) MPI_Abort(udata.comm, 1);
    retval = N_VEnableFusedOps_Raja(wsubvecs[5], opts.fusedkernels);
    if (check_flag(&retval, "N_VEnableFusedOps_Raja (main)", 1)) MPI_Abort(udata.comm, 1);
#else
    wsubvecs[5] = N_VNew_Serial(N*udata.nchem);
    if (check_flag((void *) wsubvecs[5], "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
    retval = N_VEnableFusedOps_Serial(wsubvecs[5], opts.fusedkernels);
    if (check_flag(&retval, "N_VEnableFusedOps_Serial (main)", 1)) MPI_Abort(udata.comm, 1);
#endif
    if (opts.localreduce == 0) {
      wsubvecs[5]->ops->nvdotprodlocal = NULL;
      wsubvecs[5]->ops->nvmaxnormlocal = NULL;
      wsubvecs[5]->ops->nvminlocal = NULL;
      wsubvecs[5]->ops->nvl1normlocal = NULL;
      wsubvecs[5]->ops->nvinvtestlocal = NULL;
      wsubvecs[5]->ops->nvconstrmasklocal = NULL;
      wsubvecs[5]->ops->nvminquotientlocal = NULL;
      wsubvecs[5]->ops->nvwsqrsumlocal = NULL;
      wsubvecs[5]->ops->nvwsqrsummasklocal = NULL;
    }
  }
  w = N_VNew_MPIManyVector(Nsubvecs, wsubvecs);  // combined solution vector
  if (check_flag((void *) w, "N_VNew_MPIManyVector (main)", 0)) MPI_Abort(udata.comm, 1);
  retval = N_VEnableFusedOps_MPIManyVector(w, opts.fusedkernels);
  if (check_flag(&retval, "N_VEnableFusedOps_MPIManyVector (main)", 1)) MPI_Abort(udata.comm, 1);

  // set initial conditions (or restart from file)
  if (restart < 0) {
    retval = initial_conditions(udata.t0, w, udata);
    if (check_flag(&retval, "initial_conditions (main)", 1)) MPI_Abort(udata.comm, 1);
    restart = 0;
  } else {
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
    retval = read_restart(restart, udata.t0, w, udata);
    if (check_flag(&retval, "read_restart (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  }

  //--- create the ARKStep integrator and set options ---//

  // initialize the integrator
  arkode_mem = ARKStepCreate(fEuler, NULL, udata.t0, w);
  if (check_flag((void*) arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);

  // pass udata to user functions
  retval = ARKStepSetUserData(arkode_mem, (void *) (&udata));
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  // set diagnostics file
  if (udata.showstats && outproc) {
    retval = ARKStepSetDiagnostics(arkode_mem, DFID);
    if (check_flag(&retval, "ARKStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set RK order, or specify individual Butcher table -- "order" overrides "btable"
  if (opts.order != 0) {
    retval = ARKStepSetOrder(arkode_mem, opts.order);
    if (check_flag(&retval, "ARKStepSetOrder (main)", 1)) MPI_Abort(udata.comm, 1);
  } else if (opts.btable != -1) {
    retval = ARKStepSetTableNum(arkode_mem, -1, opts.btable);
    if (check_flag(&retval, "ARKStepSetTableNum (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set dense output order
  retval = ARKStepSetDenseOrder(arkode_mem, opts.dense_order);
  if (check_flag(&retval, "ARKStepSetDenseOrder (main)", 1)) MPI_Abort(udata.comm, 1);

  // set adaptive timestepping parameters (if applicable)
  if (opts.fixedstep != 1) {

    // set safety factor
    retval = ARKStepSetSafetyFactor(arkode_mem, opts.safety);
    if (check_flag(&retval, "ARKStepSetSafetyFactor (main)", 1)) MPI_Abort(udata.comm, 1);

    // set error bias
    retval = ARKStepSetErrorBias(arkode_mem, opts.bias);
    if (check_flag(&retval, "ARKStepSetErrorBias (main)", 1)) MPI_Abort(udata.comm, 1);

    // set step growth factor
    retval = ARKStepSetMaxGrowth(arkode_mem, opts.growth);
    if (check_flag(&retval, "ARKStepSetMaxGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

    // set time step adaptivity method
    realtype adapt_params[] = {opts.k1, opts.k2, opts.k3};
    int idefault = 1;
    if (abs(opts.k1)+abs(opts.k2)+abs(opts.k3) > 0.0)  idefault=0;
    retval = ARKStepSetAdaptivityMethod(arkode_mem, opts.adapt_method, idefault,
                                        opts.pq, adapt_params);
    if (check_flag(&retval, "ARKStepSetAdaptivityMethod (main)", 1)) MPI_Abort(udata.comm, 1);

    // set first step growth factor
    retval = ARKStepSetMaxFirstGrowth(arkode_mem, opts.etamx1);
    if (check_flag(&retval, "ARKStepSetMaxFirstGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

    // set error failure growth factor
    retval = ARKStepSetMaxEFailGrowth(arkode_mem, opts.etamxf);
    if (check_flag(&retval, "ARKStepSetMaxEFailGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

    // set initial time step size
    retval = ARKStepSetInitStep(arkode_mem, opts.h0);
    if (check_flag(&retval, "ARKStepSetInitStep (main)", 1)) MPI_Abort(udata.comm, 1);

    // set minimum time step size
    retval = ARKStepSetMinStep(arkode_mem, opts.hmin);
    if (check_flag(&retval, "ARKStepSetMinStep (main)", 1)) MPI_Abort(udata.comm, 1);

    // set maximum time step size
    retval = ARKStepSetMaxStep(arkode_mem, opts.hmax);
    if (check_flag(&retval, "ARKStepSetMaxStep (main)", 1)) MPI_Abort(udata.comm, 1);

    // set maximum allowed error test failures
    retval = ARKStepSetMaxErrTestFails(arkode_mem, opts.maxnef);
    if (check_flag(&retval, "ARKStepSetMaxErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);

    // set maximum allowed hnil warnings
    retval = ARKStepSetMaxHnilWarns(arkode_mem, opts.mxhnil);
    if (check_flag(&retval, "ARKStepSetMaxHnilWarns (main)", 1)) MPI_Abort(udata.comm, 1);

    // supply cfl-stable step routine (if requested)
    if (udata.cfl > ZERO) {
      retval = ARKStepSetStabilityFn(arkode_mem, stability, (void *) (&udata));
      if (check_flag(&retval, "ARKStepSetStabilityFn (main)", 1)) MPI_Abort(udata.comm, 1);
    }

  // otherwise, set fixed timestep size
  } else {

    retval = ARKStepSetFixedStep(arkode_mem, opts.hmax);
    if (check_flag(&retval, "ARKStepSetFixedStep (main)", 1)) MPI_Abort(udata.comm, 1);

  }

  // set maximum allowed steps
  retval = ARKStepSetMaxNumSteps(arkode_mem, opts.mxsteps);
  if (check_flag(&retval, "ARKStepSetMaxNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);

  // set tolerances
  retval = ARKStepSStolerances(arkode_mem, opts.rtol, opts.atol);
  if (check_flag(&retval, "ARKStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);


  // finish initialization
  realtype t = udata.t0;
  realtype tout;
  realtype hcur;
  if (opts.dense_order == -1)
    idense = 0;
  else   // otherwise tell integrator to use dense output
    idense = 1;


  //--- Initial batch of outputs ---//
  retval = udata.profile[PR_IO].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  if (udata.myid == 0)  cout << "\nWriting initial batch of outputs\n";

  // Optionally output total mass/energy
  if (udata.showstats) {
    retval = check_conservation(udata.t0, w, udata);
    if (check_flag(&retval, "check_conservation (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // Output initial conditions to disk
  retval = output_solution(udata.t0, w, opts.h0, restart, udata, opts);
  if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);

  // Output solution statistics (if requested)
  if (udata.showstats) {
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = print_stats(t, w, 0, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // Output problem-specific diagnostic information
  retval = output_diagnostics(udata.t0, w, udata);
  if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

  // stop IO profiler
  retval = udata.profile[PR_IO].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // stop problem setup profiler
  retval = udata.profile[PR_SETUP].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  //--- Initial transient evolution: call ARKStepEvolve to perform integration   ---//
  //--- over [t0,t0+htrans], then disable adaptivity and set fixed-step size     ---//
  //--- to use for remainder of simulation.                                      ---//
  if (opts.fixedstep == 2) {

    // start transient solver profiler
    retval = udata.profile[PR_TRANS].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);

    // set transient stop time
    tout = udata.t0+opts.htrans;
    retval = ARKStepSetStopTime(arkode_mem, tout);
    if (check_flag(&retval, "ARKStepSetStopTime (main)", 1)) MPI_Abort(udata.comm, 1);

    // adaptive evolution over (t0,t0+htrans]
    retval = ARKStepEvolve(arkode_mem, tout, w, &t, ARK_NORMAL);
    if (retval < 0) {    // unsuccessful solve: break
      if (outproc)  cerr << "Solver failure, stopping integration\n";
      MPI_Abort(udata.comm, 1);
    }

    // disable adaptivity and set fixed step size
    retval = ARKStepSetFixedStep(arkode_mem, opts.hmax);
    if (check_flag(&retval, "ARKStepSetFixedStep (main)", 1)) MPI_Abort(udata.comm, 1);

    // stop transient solver profiler
    retval = udata.profile[PR_TRANS].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
  }


  //--- Main time-stepping loop: calls ARKStepEvolve to perform the integration, ---//
  //--- then prints results.  Stops when the final time has been reached.        ---//
  retval = udata.profile[PR_SIMUL].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  tout = udata.t0+dTout;
  for (int iout=restart; iout<restart+udata.nout; iout++) {

    // set stop time if applicable
    if (!idense) {
      retval = ARKStepSetStopTime(arkode_mem, tout);
      if (check_flag(&retval, "ARKStepSetStopTime (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    // evolve solution
    retval = ARKStepEvolve(arkode_mem, tout, w, &t, ARK_NORMAL);
    if (retval >= 0) {                   // successful solve: update output time
      tout = min(tout+dTout, udata.tf);
    } else {                             // unsuccessful solve: break
      if (outproc)  cerr << "Solver failure, stopping integration\n";
      MPI_Abort(udata.comm, 1);
    }

    // periodic output of solution/statistics
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

    //    output diagnostic information (if applicable)
    retval = output_diagnostics(t, w, udata);
    if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

    //    output statistics to stdout
    if (udata.showstats) {
      retval = print_stats(t, w, 1, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    //    output results to disk -- get current step from ARKStep first
    retval = ARKStepGetCurrentStep(arkode_mem, &hcur);
    if (check_flag(&retval, "ARKStepGetCurrentStep (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = output_solution(t, w, hcur, iout+1, udata, opts);
    if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  }
  if (udata.showstats) {
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
    retval = print_stats(t, w, 2, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (udata.showstats && outproc)  fclose(DFID);

  // compute simulation time
  retval = udata.profile[PR_SIMUL].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // Get some integrator statistics
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

  // Print some final statistics
  if (outproc) {
    cout << "\nFinal Solver Statistics:\n";
    cout << "   Internal solver steps = " << nst << " (attempted = " << nst_a << ")\n";
    cout << "   Total RHS evals:  Fe = " << nfe << ",  Fi = " << nfi << "\n";
    cout << "   Total number of error test failures = " << netf << "\n";
    cout << "\nProfiling Results:\n";
  }
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  udata.profile[PR_SETUP].print_cumulative_times("setup");
  udata.profile[PR_IO].print_cumulative_times("I/O");
  udata.profile[PR_MPI].print_cumulative_times("MPI");
  udata.profile[PR_PACKDATA].print_cumulative_times("pack");
  udata.profile[PR_FACEFLUX].print_cumulative_times("flux");
  udata.profile[PR_RHSEULER].print_cumulative_times("RHS");
  udata.profile[PR_DTSTAB].print_cumulative_times("dt_stab");
  udata.profile[PR_TRANS].print_cumulative_times("trans");
  udata.profile[PR_SIMUL].print_cumulative_times("sim");

  // Output mass/energy conservation error
  if (udata.showstats) {
    if (outproc)  cout << "\nConservation Check:\n";
    retval = check_conservation(t, w, udata);
    if (check_flag(&retval, "check_conservation (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // Clean up and return with successful completion
  N_VDestroy(w);               // Free solution vectors
  for (i=0; i<Nsubvecs; i++)
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  ARKStepFree(&arkode_mem);    // Free integrator memory
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  MPI_Finalize();              // Finalize MPI
  return 0;
}

//---- end of file ----
