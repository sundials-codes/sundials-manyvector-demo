/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Multirate chemistry + hydrodynamics driver:

 The "slow" time scale evolves the 3D compressible, inviscid
 Euler equations.  The "fast" time scale evolves a chemical
 network provided by Dengo -- a flexible Python library that
 creates ODE RHS and Jacobian routines for arbitrarily-complex
 chemistry networks.

 The slow time scale is evolved explicitly using ARKode's MRIStep
 time-stepping module, with it's default 3rd-order integration
 method, and a fixed time step given by either the user-input
 "initial" time step value, or calculated to equal the time
 interval between solution outputs.

 The fast time scale is evolved implicitly using ARKode's ARKStep
 time-stepping module, using the DIRK Butcher tableau that is
 specified by the user.  Here, time adaptivity is employed, with
 nearly all adaptivity options controllable via user inputs.
 Implicit subsystems are solved using the default Newton
 SUNNonlinearSolver module, but with a custom SUNLinearSolver
 module.  This is a direct solver for block-diagonal matrices
 (one block per MPI rank) that unpacks the MPIManyVector to access
 a specified subvector component (per rank), and then uses a
 standard SUNLinearSolver module for each rank-local linear
 system.  The specific SUNLinearSolver module to use on each block,
 and the MPIManyVector subvector index are provided in the module
 'constructor'.  Here, we use the KLU SUNLinearSolver module for
 the block on each rank.

 REMAINING TO-DO ITEMS:

 1. Post-processing function for steps at the slow time scale
    (PostprocessStep).  This should bring the disparate fast/slow
    energy definitions into harmony, and update the relevant
    Dengo data structures ("scale" and "inv_scale" arrays) for
    the ensuing step.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <dengo_primordial_network.hpp>
#include <arkode/arkode_mristep.h>
#include <arkode/arkode_arkstep.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>

#ifdef DEBUG
#include "fenv.h"
#endif


// Access externally-defined Dengo data structure (declared in specific
// test problem initializer), and associated initialization routine
extern cvklu_data *network_data;
int initialize_Dengo_structures(const EulerData& udata);


// user-provided functions called by the fast integrators
static int ffast(realtype t, N_Vector w, N_Vector wdot, void *user_data);
static int Jfast(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                 void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int PostprocessStep(realtype t, N_Vector y, void *user_data);

// custom Block-Diagonal MPIManyVector SUNLinearSolver module
typedef struct _BDMPIMVContent {
  SUNLinearSolver blockLS;
  sunindextype    subvec;
  sunindextype    lastflag;
} *BDMPIMVContent;
#define BDMPIMV_CONTENT(S)  ( (BDMPIMVContent)(S->content) )
#define BDMPIMV_BLS(S)      ( BDMPIMV_CONTENT(S)->blockLS )
#define BDMPIMV_SUBVEC(S)   ( BDMPIMV_CONTENT(S)->subvec )
#define BDMPIMV_LASTFLAG(S) ( BDMPIMV_CONTENT(S)->lastflag )
SUNLinearSolver SUNLinSol_BDMPIMV(SUNLinearSolver LS, N_Vector x, sunindextype subvec);
SUNLinearSolver_Type SUNLinSolGetType_BDMPIMV(SUNLinearSolver S);
int SUNLinSolInitialize_BDMPIMV(SUNLinearSolver S);
int SUNLinSolSetup_BDMPIMV(SUNLinearSolver S, SUNMatrix A);
int SUNLinSolSolve_BDMPIMV(SUNLinearSolver S, SUNMatrix A,
                           N_Vector x, N_Vector b, realtype tol);
long int SUNLinSolLastFlag_BDMPIMV(SUNLinearSolver S);



// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem parameters
  long int N, Ntot, i, j, k, l, idx;
  int Nsubvecs;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int dense_order;               // dense output order of accuracy
  int imex;                      // flag denoting class of method (0=implicit, 1=explicit, 2=IMEX)
  int fixedpt;                   // flag denoting use of fixed-point nonlinear solver
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use (disabled if negative)
  N_Vector w = NULL;             // empty vectors for storing overall solution
  N_Vector *wsubvecs;
  N_Vector atols = NULL;
  SUNMatrix A = NULL;            // empty matrix and linear solver structures
  SUNLinearSolver LS = NULL;
  SUNLinearSolver BLS = NULL;
  void *outer_arkode_mem = NULL; // empty ARKStep memory structures
  void *inner_arkode_mem = NULL;

  //--- General Initialization ---//

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

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

  // set slow timestep size as h0 (if >0), or dTout otherwise
  realtype hslow = (opts.h0 > 0) ? opts.h0 : dTout;

  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);

  // ensure that this was compiled with chemical species
  if (udata.nchem == 0) {
    if (udata.myid == 0)
      cerr << "\nError: executable <must> be compiled with chemical species enabled\n";
    MPI_Abort(udata.comm, 1);
  }

  // Output problem setup information
  bool outproc = (udata.myid == 0);
  if (outproc) {
    cout << "\n3D compressible inviscid Euler + primordial chemistry driver:\n";
    cout << "   nprocs: " << udata.nprocs << " (" << udata.npx << " x "
         << udata.npy << " x " << udata.npz << ")\n";
    cout << "   spatial domain: [" << udata.xl << ", " << udata.xr << "] x ["
         << udata.yl << ", " << udata.yr << "] x ["
         << udata.zl << ", " << udata.zr << "]\n";
    cout << "   time domain = (" << udata.t0 << ", " << udata.tf << "]\n";
    cout << "   slow timestep size: " << hslow << "\n";
    cout << "   bdry cond (" << BC_PERIODIC << "=per, " << BC_NEUMANN << "=Neu, "
         << BC_DIRICHLET << "=Dir, " << BC_REFLECTING << "=refl): ["
         << udata.xlbc << ", " << udata.xrbc << "] x ["
         << udata.ylbc << ", " << udata.yrbc << "] x ["
         << udata.zlbc << ", " << udata.zrbc << "]\n";
    cout << "   gamma: " << udata.gamma << "\n";
    cout << "   num chemical species: " << udata.nchem << "\n";
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

  // open solver diagnostics output files for writing
  FILE *DFID_OUTER = NULL;
  FILE *DFID_INNER = NULL;
  if (outproc) {
    DFID_OUTER=fopen("diags_hydro.txt","w");
    DFID_INNER=fopen("diags_chem.txt","w");
  }

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
  atols = N_VClone(w);                           // absolute tolerance vector for fast stepper
  if (check_flag((void *) atols, "N_VClone (main)", 0)) MPI_Abort(udata.comm, 1);
  N_VConst(opts.atol, atols);

  // initialize Dengo data structure, "network_data"
  retval = initialize_Dengo_structures(udata);
  if (check_flag(&retval, "initialize_Dengo_structures (main)", 1)) MPI_Abort(udata.comm, 1);

  // set initial conditions (or restart from file)
  if (restart < 0) {
    retval = initial_conditions(udata.t0, w, udata);
    if (check_flag(&retval, "initial_conditions (main)", 1)) MPI_Abort(udata.comm, 1);
    restart = 0;
  } else {
    retval = read_restart(restart, udata.t0, w, udata);
    if (check_flag(&retval, "read_restart (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // move input chemical solution values into 'scale' components of network_data structure
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (main)", 0)) return -1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++)
        for (l=0; l<udata.nchem; l++) {
          idx = BUFIDX(l,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
          network_data->scale[0][idx] = chem[idx];
          network_data->inv_scale[0][idx] = ONE / chem[idx];
          chem[idx] = ONE;
        }

  // compute auxiliary values within network_data structure
  setting_up_extra_variables(network_data, network_data->scale[0], N);


  //--- create the fast integrator and set options ---//

  // initialize the fast integrator.
  inner_arkode_mem = ARKStepCreate(NULL, ffast, udata.t0, w);
  if (check_flag((void*) inner_arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);

  // create the fast integrator linear solver module, and attach to ARKStep
  A  = SUNSparseMatrix(N, N, 64*N, CSR_MAT);
  if (check_flag((void*) A, "SUNSparseMatrix (main)", 0)) MPI_Abort(udata.comm, 1);
  BLS = SUNLinSol_KLU(wsubvecs[5], A);
  if (check_flag((void*) LS, "SUNLinSol_KLU (main)", 0)) MPI_Abort(udata.comm, 1);
  LS = SUNLinSol_BDMPIMV(BLS, w, 5);
  if (check_flag((void*) LS, "SUNLinSol_BDMPIMV (main)", 0)) MPI_Abort(udata.comm, 1);
  retval = ARKStepSetLinearSolver(inner_arkode_mem, LS, A);
  if (check_flag(&retval, "ARKStepSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepSetJacFn(inner_arkode_mem, Jfast);
  if (check_flag(&retval, "ARKStepSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);

  // pass network_udata to user functions
  retval = ARKStepSetUserData(inner_arkode_mem, network_data);
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  // set diagnostics file
  if (outproc) {
    retval = ARKStepSetDiagnostics(inner_arkode_mem, DFID_INNER);
    if (check_flag(&retval, "ARKStepSetDiagnostics (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set inner RK Butcher table
  if (opts.btable != -1) {
    retval = ARKStepSetTableNum(inner_arkode_mem, opts.btable, -1);
    if (check_flag(&retval, "ARKStepSetTableNum (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set safety factor
  retval = ARKStepSetSafetyFactor(inner_arkode_mem, opts.safety);
  if (check_flag(&retval, "ARKStepSetSafetyFactor (main)", 1)) MPI_Abort(udata.comm, 1);

  // set error bias
  retval = ARKStepSetErrorBias(inner_arkode_mem, opts.bias);
  if (check_flag(&retval, "ARKStepSetErrorBias (main)", 1)) MPI_Abort(udata.comm, 1);

  // set step growth factor
  retval = ARKStepSetMaxGrowth(inner_arkode_mem, opts.growth);
  if (check_flag(&retval, "ARKStepSetMaxGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

  // set time step adaptivity method
  realtype adapt_params[] = {opts.k1, opts.k2, opts.k3};
  int idefault = 1;
  if (abs(opts.k1)+abs(opts.k2)+abs(opts.k3) > 0.0)  idefault=0;
  retval = ARKStepSetAdaptivityMethod(inner_arkode_mem, opts.adapt_method,
                                      idefault, opts.pq, adapt_params);
  if (check_flag(&retval, "ARKStepSetAdaptivityMethod (main)", 1)) MPI_Abort(udata.comm, 1);

  // set first step growth factor
  retval = ARKStepSetMaxFirstGrowth(inner_arkode_mem, opts.etamx1);
  if (check_flag(&retval, "ARKStepSetMaxFirstGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

  // set error failure growth factor
  retval = ARKStepSetMaxEFailGrowth(inner_arkode_mem, opts.etamxf);
  if (check_flag(&retval, "ARKStepSetMaxEFailGrowth (main)", 1)) MPI_Abort(udata.comm, 1);

  // set minimum time step size
  retval = ARKStepSetMinStep(inner_arkode_mem, opts.hmin);
  if (check_flag(&retval, "ARKStepSetMinStep (main)", 1)) MPI_Abort(udata.comm, 1);

  // set maximum time step size
  retval = ARKStepSetMaxStep(inner_arkode_mem, opts.hmax);
  if (check_flag(&retval, "ARKStepSetMaxStep (main)", 1)) MPI_Abort(udata.comm, 1);

  // set maximum allowed error test failures
  retval = ARKStepSetMaxErrTestFails(inner_arkode_mem, opts.maxnef);
  if (check_flag(&retval, "ARKStepSetMaxErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);

  // set maximum allowed hnil warnings
  retval = ARKStepSetMaxHnilWarns(inner_arkode_mem, opts.mxhnil);
  if (check_flag(&retval, "ARKStepSetMaxHnilWarns (main)", 1)) MPI_Abort(udata.comm, 1);

  // set maximum allowed steps
  retval = ARKStepSetMaxNumSteps(inner_arkode_mem, opts.mxsteps);
  if (check_flag(&retval, "ARKStepSetMaxNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);

  // set tolerances
  retval = ARKStepSVtolerances(inner_arkode_mem, opts.rtol, atols);
  if (check_flag(&retval, "ARKStepSVtolerances (main)", 1)) MPI_Abort(udata.comm, 1);

  // set implicit predictor method
  retval = ARKStepSetPredictorMethod(inner_arkode_mem, opts.predictor);
  if (check_flag(&retval, "ARKStepSetPredictorMethod (main)", 1)) MPI_Abort(udata.comm, 1);

  // set max nonlinear iterations
  retval = ARKStepSetMaxNonlinIters(inner_arkode_mem, opts.maxniters);
  if (check_flag(&retval, "ARKStepSetMaxNonlinIters (main)", 1)) MPI_Abort(udata.comm, 1);

  // set nonlinear tolerance safety factor
  retval = ARKStepSetNonlinConvCoef(inner_arkode_mem, opts.nlconvcoef);
  if (check_flag(&retval, "ARKStepSetNonlinConvCoef (main)", 1)) MPI_Abort(udata.comm, 1);


  //--- create the slow integrator and set options ---//

  // initialize the integrator memory
  outer_arkode_mem = MRIStepCreate(fEuler, udata.t0, w, MRISTEP_ARKSTEP, inner_arkode_mem);
  if (check_flag((void*) outer_arkode_mem, "MRIStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);

  // pass udata to user functions
  retval = MRIStepSetUserData(outer_arkode_mem, (void *) (&udata));
  if (check_flag(&retval, "MRIStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  // set diagnostics file
  if (outproc) {
    retval = MRIStepSetDiagnostics(outer_arkode_mem, DFID_OUTER);
    if (check_flag(&retval, "MRIStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set slow time step size
  retval = MRIStepSetFixedStep(outer_arkode_mem, hslow);
  if (check_flag(&retval, "MRIStepSetFixedStep (main)", 1)) MPI_Abort(udata.comm, 1);


  //--- Initial batch of outputs ---//

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

  // compute overall setup time
  double tsetup = MPI_Wtime() - tstart;
  tstart = MPI_Wtime();

  /* Main time-stepping loop: calls MRIStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  realtype t = udata.t0;
  realtype tout = udata.t0+dTout;
  realtype hcur;
  if (udata.showstats) {
    retval = print_stats(t, w, 0, outer_arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  int iout;
  for (iout=restart; iout<restart+udata.nout; iout++) {

    retval = MRIStepEvolve(outer_arkode_mem, tout, w, &t, ARK_NORMAL);  // call integrator
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
      retval = print_stats(t, w, 1, outer_arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    // output diagnostic information (if applicable)
    retval = output_diagnostics(t, w, udata);
    if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

    // output results to disk -- get current step from MRIStep first
    retval = MRIStepGetLastStep(outer_arkode_mem, &hcur);
    if (check_flag(&retval, "MRIStepGetLastStep (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = output_solution(t, w, hcur, iout+1, udata, opts);
    if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
    tinout += MPI_Wtime() - iostart;

  }
  if (udata.showstats) {
    iostart = MPI_Wtime();
    retval = print_stats(t, w, 2, outer_arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    tinout += MPI_Wtime() - iostart;
  }
  if (outproc) {
    fclose(DFID_OUTER);
    fclose(DFID_INNER);
  }

  // compute simulation time
  double tsimul = MPI_Wtime() - tstart;

  // Get some slow integrator statistics
  long int nsts, nfs;
  nsts = nfs = 0;
  retval = MRIStepGetNumSteps(outer_arkode_mem, &nsts);
  if (check_flag(&retval, "MRIStepGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = MRIStepGetNumRhsEvals(outer_arkode_mem, &nfs);
  if (check_flag(&retval, "MRIStepGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = MRIStepSetPostprocessStepFn(outer_arkode_mem, PostprocessStep);

  // Get some fast integrator statistics
  long int nstf, nstf_a, nfe, nfi, netf, nls, nni, ncf, nje;
  nstf = nstf_a = nfe = nfi = netf = nls = nni = ncf = nje = 0;
  retval = ARKStepGetNumSteps(inner_arkode_mem, &nstf);
  if (check_flag(&retval, "ARKStepGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumStepAttempts(inner_arkode_mem, &nstf_a);
  if (check_flag(&retval, "ARKStepGetNumStepAttempts (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumRhsEvals(inner_arkode_mem, &nfe, &nfi);
  if (check_flag(&retval, "ARKStepGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumErrTestFails(inner_arkode_mem, &netf);
  if (check_flag(&retval, "ARKStepGetNumErrTestFails (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinSolvSetups(inner_arkode_mem, &nls);
  if (check_flag(&retval, "ARKStepGetNumLinSolvSetups (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNonlinSolvStats(inner_arkode_mem, &nni, &ncf);
  if (check_flag(&retval, "ARKStepGetNonlinSolvStats (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumJacEvals(inner_arkode_mem, &nje);
  if (check_flag(&retval, "ARKStepGetNumJacEvals (main)", 1)) MPI_Abort(udata.comm, 1);

  // Print some final statistics
  if (outproc) {
    cout << "\nFinal Solver Statistics:\n";
    cout << "   Slow solver steps = " << nsts << "\n";
    cout << "   Fast solver steps = " << nstf << " (attempted = " << nstf_a << ")\n";
    cout << "   Total RHS evals:  Fs = " << nfs << ",  Ff = " << nfi << "\n";
    cout << "   Total number of fast error test failures = " << netf << "\n";
    if (nls > 0)
      cout << "   Total number of fast lin solv setups = " << nls << "\n";
    if (nls > 0)
      cout << "   Total number of fast lin solv setups = " << nls << "\n";
    if (nni > 0) {
      cout << "   Total number of fast nonlin iters = " << nni << "\n";
      cout << "   Total number of fast nonlin conv fails = " << ncf << "\n";
    }
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
  MRIStepFree(&outer_arkode_mem);  // Free integrator memory
  ARKStepFree(&inner_arkode_mem);
  SUNLinSolFree(BLS);              // Free matrix and linear solvers
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(w);                   // Free solution/tolerance vectors
  for (i=0; i<Nsubvecs; i++)
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  N_VDestroy(atols);
  MPI_Finalize();                  // Finalize MPI
  return 0;
}


//---- fast/slow problem-defining functions (wrappers for other routines) ----

static int ffast(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // unpack chemistry subvectors, and call Dengo RHS routine
  N_Vector wchem = NULL;
  wchem = N_VGetSubvector_MPIManyVector(w, 5);
  if (check_flag((void *) wchem, "N_VGetSubvector_MPIManyVector (ffast)", 0)) return(1);
  N_Vector wchemdot = NULL;
  wchemdot = N_VGetSubvector_MPIManyVector(wdot, 5);
  if (check_flag((void *) wchemdot, "N_VGetSubvector_MPIManyVector (ffast)", 0)) return(1);
  return( calculate_rhs_cvklu(t, wchem, wchemdot, user_data) );
}

static int Jfast(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                 void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // unpack chemistry subvectors, and call Dengo Jacobian routine
  N_Vector wchem = NULL;
  wchem = N_VGetSubvector_MPIManyVector(w, 5);
  if (check_flag((void *) wchem, "N_VGetSubvector_MPIManyVector (Jfast)", 0)) return(1);
  N_Vector fwchem = NULL;
  fwchem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) fwchem, "N_VGetSubvector_MPIManyVector (Jfast)", 0)) return(1);
  N_Vector tmp1chem = NULL;
  tmp1chem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) tmp1chem, "N_VGetSubvector_MPIManyVector (Jfast)", 0)) return(1);
  N_Vector tmp2chem = NULL;
  tmp2chem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) tmp2chem, "N_VGetSubvector_MPIManyVector (Jfast)", 0)) return(1);
  N_Vector tmp3chem = NULL;
  tmp3chem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) tmp3chem, "N_VGetSubvector_MPIManyVector (Jfast)", 0)) return(1);
  return( calculate_sparse_jacobian_cvklu(t, wchem, fwchem, Jac, user_data,
                                          tmp1chem, tmp2chem, tmp3chem) );
}


//---- slow step post-processing function ----
// This routine brings the disparate fast/slow energy definitions into harmony, and
// updates the relevant Dengo data structures for the ensuing step

static int PostprocessStep(realtype t, N_Vector y, void *user_data)
{

  //--- FILL THIS IN --//

  return(0);
}


//---- custom SUNLinearSolver module ----

SUNLinearSolver SUNLinSol_BDMPIMV(SUNLinearSolver BLS, N_Vector x, sunindextype subvec)
{
  SUNLinearSolver S;
  BDMPIMVContent content;

  // Check compatibility with supplied N_Vector
  if (N_VGetVectorID(x) != SUNDIALS_NVEC_MPIMANYVECTOR) return(NULL);
  if (subvec >= N_VGetNumSubvectors_MPIManyVector(x)) return(NULL);

  // Create an empty linear solver
  S = NULL;
  S = SUNLinSolNewEmpty();
  if (S == NULL) return(NULL);

  // Attach operations (use defaults whenever possible)
  S->ops->gettype    = SUNLinSolGetType_BDMPIMV;
  S->ops->initialize = SUNLinSolInitialize_BDMPIMV;
  S->ops->setup      = SUNLinSolSetup_BDMPIMV;
  S->ops->solve      = SUNLinSolSolve_BDMPIMV;
  S->ops->lastflag   = SUNLinSolLastFlag_BDMPIMV;

  // Create, fill and attach content
  content = NULL;
  content = (BDMPIMVContent) malloc(sizeof *content);
  if (content == NULL) { SUNLinSolFree(S); return(NULL); }
  content->blockLS = BLS;
  content->subvec  = subvec;
  S->content = content;

  return(S);
}

SUNLinearSolver_Type SUNLinSolGetType_BDMPIMV(SUNLinearSolver S)
{
  return(SUNLINEARSOLVER_DIRECT);
}

int SUNLinSolInitialize_BDMPIMV(SUNLinearSolver S)
{
  // pass initialize call down to block linear solver
  BDMPIMV_LASTFLAG(S) = SUNLinSolInitialize(BDMPIMV_BLS(S));
  return(BDMPIMV_LASTFLAG(S));
}

int SUNLinSolSetup_BDMPIMV(SUNLinearSolver S, SUNMatrix A)
{
  // pass setup call down to block linear solver
  BDMPIMV_LASTFLAG(S) = SUNLinSolSetup(BDMPIMV_BLS(S), A);
  return(BDMPIMV_LASTFLAG(S));
}

int SUNLinSolSolve_BDMPIMV(SUNLinearSolver S, SUNMatrix A,
                           N_Vector x, N_Vector b, realtype tol)
{
  // access desired subvector from MPIManyVector objects
  N_Vector xsub, bsub;
  xsub = bsub = NULL;
  xsub = N_VGetSubvector_MPIManyVector(x, BDMPIMV_SUBVEC(S));
  bsub = N_VGetSubvector_MPIManyVector(b, BDMPIMV_SUBVEC(S));
  if ((xsub == NULL) || (bsub == NULL)) {
    BDMPIMV_LASTFLAG(S) = SUNLS_MEM_FAIL;
    return(BDMPIMV_LASTFLAG(S));
  }

  // pass solve call down to block linear solver
  BDMPIMV_LASTFLAG(S) = SUNLinSolSolve(BDMPIMV_BLS(S), A,
                                       xsub, bsub, tol);
  return(BDMPIMV_LASTFLAG(S));
}

long int SUNLinSolLastFlag_BDMPIMV(SUNLinearSolver S)
{
  return(BDMPIMV_LASTFLAG(S));
}


//---- end of file ----
