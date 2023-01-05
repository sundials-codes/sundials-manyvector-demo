/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
                David J. Gardner @ LLNL
 ----------------------------------------------------------------
 Copyright (c) 2022, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Multirate chemistry + hydrodynamics driver:

 The "slow" time scale evolves the 3D compressible, inviscid
 Euler equations.  The "fast" time scale evolves a chemical
 network provided by Dengo -- a flexible Python library that
 creates ODE RHS and Jacobian routines for arbitrarily-complex
 chemistry networks.

 The slow time scale is evolved explicitly using ARKODE's MRIStep
 time-stepping module, with it's default 3rd-order integration
 method, and a fixed time step given by either the user-input
 "initial" time step value, h0, or calculated to equal the time
 interval between solution outputs.

 The fast time scale is evolved implicitly using ARKODE's ARKStep
 time-stepping module, using the DIRK Butcher tableau that is
 specified by the user.  Here, time adaptivity is employed, with
 nearly all adaptivity options controllable via user inputs.
 If the input file specifies fixedstep=1, then temporal adaptivity
 is disabled, and the solver will use the fixed step size
 h=hmax.  In this case, if the input file specifies htrans>0,
 then temporal adaptivity will be used for the start of the
 simulation [t0,t0+htrans], followed by fixed time-stepping using
 h=hmax.  We require that htrans is smaller than the first
 output time interval, i.e., t0+htrans < t0+dTout.  Implicit
 subsystems are solved using the default Newton SUNNonlinearSolver
 module, but with a custom SUNLinearSolver module.  This is a
 direct solver for block-diagonal matrices (one block per MPI
 rank) that unpacks the MPIManyVector to access a specified
 subvector component (per rank), and then uses a standard
 SUNLinearSolver module for each rank-local linear system.  The
 specific SUNLinearSolver module to use on each block, and the
 MPIManyVector subvector index are provided in the module
 'constructor'.  Here, we use the KLU SUNLinearSolver module for
 the block on each rank.
 ---------------------------------------------------------------*/

// Header files
//    Physics
#include <euler3D.hpp>
#include <raja_primordial_network.hpp>

//    SUNDIALS
#include <arkode/arkode_mristep.h>
#include <arkode/arkode_arkstep.h>
#ifdef USE_DEVICE
#include <sunmatrix/sunmatrix_magmadense.h>
#include <sunlinsol/sunlinsol_magmadense.h>
#else
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#endif

#ifdef DEBUG
#include "fenv.h"
#endif

//#define DISABLE_HYDRO
//#define SETUP_ONLY

// macros for handling formatting of diagnostic output
#define PRINT_CGS 1
#define PRINT_SCIENTIFIC 1



// Initialization and preparation routines for Dengo data structure
// (provided in specific test problem initializer)
int initialize_Dengo_structures(EulerData& udata);
void free_Dengo_structures(EulerData& udata);
int prepare_Dengo_structures(realtype& t, N_Vector w, EulerData& udata);
int apply_Dengo_scaling(N_Vector w, EulerData& udata);
int unapply_Dengo_scaling(N_Vector w, EulerData& udata);


// user-provided functions called by the fast integrators
static int ffast(realtype t, N_Vector w, N_Vector wdot, void* user_data);
static int Jfast(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                 void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int fslow(realtype t, N_Vector w, N_Vector wdot, void* user_data);


// custom MPI rank-local fast integrator
struct RankLocalStepperContent {
   void* solver_mem = NULL;              // local integrator memory structure
   MRIStepInnerStepper stepper = NULL;   // inner stepper memory structure
   EulerData* udata = NULL;              // user data pointer
   bool disable_forcing = false;         // flag to disable forcing in RHS
};
static int RankLocalStepper_Evolve(MRIStepInnerStepper stepper, realtype t0,
                                   realtype tout, N_Vector y);
static int RankLocalStepper_FullRhs(MRIStepInnerStepper stepper, realtype t,
                                   N_Vector y, N_Vector f, int mode);
static int RankLocalStepper_Reset(MRIStepInnerStepper stepper, realtype tR,
                                   N_Vector yR);
static int RankLocalStepper_GetStats(void* inner_arkode_mem, EulerData& udata,
                                     SUNLinearSolver& ranklocalLS,
                                     long int& nstf_max, long int& nstf_min,
                                     long int& nstf_a_max, long int& nstf_a_min,
                                     long int& nfi_max, long int& nfi_min,
                                     long int& netf_max, long int& netf_min,
                                     long int& nni_max, long int& nni_min,
                                     long int& ncf_max, long int& ncf_min,
                                     long int& nls_max, long int& nls_min,
                                     long int& nje_max, long int& nje_min,
                                     long int& nli_max, long int& nli_min,
                                     long int& nlcf_max, long int& nlcf_min,
                                     long int& nlfs_max, long int& nlfs_min);

// custom rank-local chemistry SUNLinearSolver module
typedef struct _RankLocalLSContent {
  SUNLinearSolver blockLS;
  sunindextype    subvec;
  sunindextype    lastflag;
  EulerData*      udata;
  void*           arkode_mem;
  long int        nfeDQ;
} *RankLocalLSContent;
#define RankLocalLS_CONTENT(S)  ( (RankLocalLSContent)(S->content) )
#define RankLocalLS_BLS(S)      ( RankLocalLS_CONTENT(S)->blockLS )
#define RankLocalLS_SUBVEC(S)   ( RankLocalLS_CONTENT(S)->subvec )
#define RankLocalLS_LASTFLAG(S) ( RankLocalLS_CONTENT(S)->lastflag )
#define RankLocalLS_UDATA(S)    ( RankLocalLS_CONTENT(S)->udata )
#define RankLocalLS_NFEDQ(S)    ( RankLocalLS_CONTENT(S)->nfeDQ )
SUNLinearSolver SUNLinSol_RankLocalLS(SUNLinearSolver LS, N_Vector x,
                                  sunindextype subvec, EulerData* udata,
                                  void *arkode_mem, ARKODEParameters& opts,
                                  SUNContext ctx);
SUNLinearSolver_Type GetType_RankLocalLS(SUNLinearSolver S);
int Initialize_RankLocalLS(SUNLinearSolver S);
int Setup_RankLocalLS(SUNLinearSolver S, SUNMatrix A);
int Solve_RankLocalLS(SUNLinearSolver S, SUNMatrix A,
                      N_Vector x, N_Vector b, realtype tol);
sunindextype LastFlag_RankLocalLS(SUNLinearSolver S);
int Free_RankLocalLS(SUNLinearSolver S);


// utility routines
void cleanup(void **outer_arkode_mem, void **inner_arkode_mem,
             MRIStepInnerStepper stepper,
             RankLocalStepperContent *inner_content, EulerData& udata,
             SUNLinearSolver BLS, SUNLinearSolver LS, SUNMatrix A, N_Vector w,
             N_Vector wloc, N_Vector atols, N_Vector *wsubvecs, int Nsubvecs);


// Main Program
int main(int argc, char* argv[]) {

  // initialize MPI
  int myid, retval;
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return(1);
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem variables
  long int N;
  int Nsubvecs;
  int restart;                   // restart file number to use (disabled if negative)
  int NoOutput;                  // flag for case when no output is desired
  N_Vector w = NULL;             // empty vectors
  N_Vector wloc = NULL;
  N_Vector *wsubvecs = NULL;
  N_Vector atols = NULL;
  SUNMatrix A = NULL;            // empty matrix and linear solver structures
  SUNLinearSolver LS = NULL;
  SUNLinearSolver BLS = NULL;
  MRIStepInnerStepper stepper = NULL;
  void *outer_arkode_mem = NULL; // empty ARKStep memory structures
  void *inner_arkode_mem = NULL;
  EulerData udata;               // solver data structures
  ARKODEParameters opts;

  //--- General Initialization ---//

  // start various code profilers
  retval = udata.profile[PR_TOTAL].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  retval = udata.profile[PR_SETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  retval = udata.profile[PR_IO].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  if (myid == 0)  cout << "Initializing problem\n";

  // read problem and solver parameters from input file / command line
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  if (myid == 0)  cout << "Setting up parallel decomposition\n";

  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);
  bool outproc = (udata.myid == 0);

  // set NoOutput flag based on nout input
  NoOutput = 0;
  if (udata.nout <= 0) {
    NoOutput = 1;
    udata.nout = 1;
  }

  // set output time frequency
  realtype dTout = (udata.tf-udata.t0)/udata.nout;
  retval = udata.profile[PR_IO].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

  // set slow timestep size as h0 (if >0), or dTout otherwise
  realtype hslow = (opts.h0 > 0) ? opts.h0 : dTout;

  // if fixed time stepping is specified, ensure that hmax>0
  if (opts.fixedstep && (opts.hmax <= ZERO)) {
    if (outproc)  cerr << "\nError: fixed time stepping requires hmax > 0 ("
                       << opts.hmax << " given)\n";
    MPI_Abort(udata.comm, 1);
  }

  // update fixedstep parameter when initial transient evolution is requested
  if (opts.fixedstep && (opts.htrans>0))  opts.fixedstep=2;

  // ensure that htrans < dTout
  if (opts.htrans >= dTout) {
    if (outproc)  cerr << "\nError: htrans (" << opts.htrans << ") >= dTout (" << dTout << ")\n";
    MPI_Abort(udata.comm, 1);
  }

  // ensure that this was compiled with chemical species
  if (udata.nchem == 0) {
    if (outproc)  cerr << "\nError: executable <must> be compiled with chemical species enabled\n";
    MPI_Abort(udata.comm, 1);
  }

  // Output problem setup information
  if (outproc) {
    cout << "\n3D compressible inviscid Euler + primordial chemistry driver (multirate):\n";
    cout << "   nprocs: " << udata.nprocs << " (" << udata.npx << " x "
         << udata.npy << " x " << udata.npz << ")\n";
    cout << "   spatial domain: [" << udata.xl << ", " << udata.xr << "] x ["
         << udata.yl << ", " << udata.yr << "] x ["
         << udata.zl << ", " << udata.zr << "]\n";
    cout << "   time domain = (" << udata.t0 << ", " << udata.tf << "]"
         << ",  or (" << udata.t0*udata.TimeUnits
         << ", " << udata.tf*udata.TimeUnits << "] in CGS\n";
    cout << "   slow timestep size: " << hslow << "\n";
    if (opts.fixedstep > 0)
      cout << "   fixed timestep size: " << opts.hmax << "\n";
    if (opts.fixedstep == 2)
      cout << "   initial transient evolution: " << opts.htrans << "\n";
    if (NoOutput == 1) {
      cout << "   solution output disabled\n";
    } else {
      cout << "   output timestep size: " << dTout << "\n";
    }
    cout << "   bdry cond (" << BC_PERIODIC << "=per, " << BC_NEUMANN << "=Neu, "
         << BC_DIRICHLET << "=Dir, " << BC_REFLECTING << "=refl): ["
         << udata.xlbc << ", " << udata.xrbc << "] x ["
         << udata.ylbc << ", " << udata.yrbc << "] x ["
         << udata.zlbc << ", " << udata.zrbc << "]\n";
    cout << "   gamma: " << udata.gamma << "\n";
    cout << "   num chemical species: " << udata.nchem << "\n";
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
#ifdef DISABLE_HYDRO
    cout << "Hydrodynamics is turned OFF\n";
#endif
#if defined(RAJA_CUDA)
    cout << "Executable built with RAJA+CUDA support and MAGMA linear solver\n";
#elif defined(RAJA_HIP)
    cout << "Executable built with RAJA+HIP support and MAGMA linear solver\n";
#elif defined(RAJA_OPENMP)
    cout << "Executable built with RAJA+OpenMP support and KLU linear solver\n";
#else
    cout << "Executable built with RAJA+SERIAL support and KLU linear solver\n";
#endif
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

  // open solver diagnostics output files for writing
  FILE *DFID_OUTER = NULL;
  FILE *DFID_INNER = NULL;
  if (udata.showstats && outproc) {
    cout << "Creating diagnostics output files\n";
    DFID_OUTER=fopen("diags_hydro.txt","w");
    DFID_INNER=fopen("diags_chem.txt","w");
  }

  // Initialize N_Vector data structures with configured vector operations
  N = (udata.nxl)*(udata.nyl)*(udata.nzl);
  Nsubvecs = 5 + ((udata.nchem > 0) ? 1 : 0);
  wsubvecs = new N_Vector[Nsubvecs];
  for (int i=0; i<5; i++) {
    wsubvecs[i] = NULL;
    wsubvecs[i] = N_VNew_Serial(N, udata.ctx);
    if (check_flag((void *) wsubvecs[i], "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
    retval = N_VEnableFusedOps_Serial(wsubvecs[i], opts.fusedkernels);
    if (check_flag(&retval, "N_VEnableFusedOps_Serial (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  if (udata.nchem > 0) {
    wsubvecs[5] = NULL;
#ifdef USE_DEVICE
    wsubvecs[5] = N_VNewManaged_Raja(N*udata.nchem, udata.ctx);
    if (check_flag((void *) wsubvecs[5], "N_VNewManaged_Raja (main)", 0)) MPI_Abort(udata.comm, 1);
    retval = N_VEnableFusedOps_Raja(wsubvecs[5], opts.fusedkernels);
    if (check_flag(&retval, "N_VEnableFusedOps_Raja (main)", 1)) MPI_Abort(udata.comm, 1);
#else
    wsubvecs[5] = N_VNew_Serial(N*udata.nchem, udata.ctx);
    if (check_flag((void *) wsubvecs[5], "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
    retval = N_VEnableFusedOps_Serial(wsubvecs[5], opts.fusedkernels);
    if (check_flag(&retval, "N_VEnableFusedOps_Serial (main)", 1)) MPI_Abort(udata.comm, 1);
#endif
  }
  w = N_VMake_MPIManyVector(udata.comm, Nsubvecs, wsubvecs, udata.ctx);  // combined solution vector
  if (check_flag((void *) w, "N_VMake_MPIManyVector (main)", 0)) MPI_Abort(udata.comm, 1);
  retval = N_VEnableFusedOps_MPIManyVector(w, opts.fusedkernels);
  if (check_flag(&retval, "N_VEnableFusedOps_MPIManyVector (main)", 1)) MPI_Abort(udata.comm, 1);
  wloc = N_VNew_ManyVector(Nsubvecs, wsubvecs, udata.ctx);  // rank-local solution vector (fast stepper)
  if (check_flag((void *) wloc, "N_VNew_ManyVector (main)", 0)) MPI_Abort(udata.comm, 1);
  retval = N_VEnableFusedOps_ManyVector(wloc, opts.fusedkernels);
  if (check_flag(&retval, "N_VEnableFusedOps_ManyVector (main)", 1)) MPI_Abort(udata.comm, 1);
  atols = N_VClone(wloc);                             // absolute tolerance vector (fast stepper)
  if (check_flag((void *) atols, "N_VClone (main)", 0)) MPI_Abort(udata.comm, 1);
  N_VConst(opts.atol, atols);

  // initialize Dengo data structure, "network_data" (stored within udata)
  retval = initialize_Dengo_structures(udata);
  if (check_flag(&retval, "initialize_Dengo_structures (main)", 1)) MPI_Abort(udata.comm, 1);

  // set initial conditions into overall solution vector (or restart from file)
  // [note: since w and wloc share the same component N_Vectors, this also initializes wloc]
  if (restart < 0) {
    retval = initial_conditions(udata.t0, w, udata);
    if (check_flag(&retval, "initial_conditions (main)", 1)) MPI_Abort(udata.comm, 1);
    restart = 0;
  } else {
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = read_restart(restart, udata.t0, w, udata);
    if (check_flag(&retval, "read_restart (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // prepare Dengo structures and initial condition vector(s) for fast time scale evolution
  retval = prepare_Dengo_structures(udata.t0, w, udata);
  if (check_flag(&retval, "prepare_Dengo_structures (main)", 1)) MPI_Abort(udata.comm, 1);


  //--- create the fast integrator and set options ---//

  // create MRIStepInnerStepper wrapper for fast integrator, and set functions
  retval = MRIStepInnerStepper_Create(udata.ctx, &stepper);
  if (check_flag(&retval, "MRIStepInnerStepper_Create (main)", 1)) MPI_Abort(udata.comm, 1);
  RankLocalStepperContent *inner_content = new RankLocalStepperContent;
  inner_content->udata = &udata;
  inner_content->stepper = stepper;
  retval = MRIStepInnerStepper_SetContent(stepper, inner_content);
  if (check_flag(&retval, "MRIStepInnerStepper_SetContent", 1)) return 1;
  retval = MRIStepInnerStepper_SetEvolveFn(stepper, RankLocalStepper_Evolve);
  if (check_flag(&retval, "MRIStepInnerStepper_SetEvolve", 1)) return 1;
  retval = MRIStepInnerStepper_SetFullRhsFn(stepper, RankLocalStepper_FullRhs);
  if (check_flag(&retval, "MRIStepInnerStepper_SetFullRhsFn", 1)) return 1;
  retval = MRIStepInnerStepper_SetResetFn(stepper, RankLocalStepper_Reset);
  if (check_flag(&retval, "MRIStepInnerStepper_SetResetFn", 1)) return 1;

  // initialize the fast integrator using only the chemistry subvector,
  // and attach to MRIStepInnerStepper content
  inner_arkode_mem = ARKStepCreate(NULL, ffast, udata.t0, wloc, udata.ctx);
  if (check_flag((void*) inner_arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);
  inner_content->solver_mem = inner_arkode_mem;

  // pass inner_content to fast integrator user functions
  retval = ARKStepSetUserData(inner_arkode_mem, (void*) inner_content);
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  // create the fast integrator local linear solver
#ifdef USE_DEVICE
  // Create SUNMatrix for use in linear solves
  A = SUNMatrix_MagmaDenseBlock(N, udata.nchem, udata.nchem, SUNMEMTYPE_DEVICE,
                                udata.memhelper, NULL, udata.ctx);
  if(check_flag((void *) A, "SUNMatrix_MagmaDenseBlock", 0)) return(1);
#else
  A = SUNSparseMatrix(N*udata.nchem, N*udata.nchem, 64*N*udata.nchem, CSR_MAT, udata.ctx);
  if (check_flag((void*) A, "SUNSparseMatrix (main)", 0)) MPI_Abort(udata.comm, 1);
#endif

  // Create the custom SUNLinearSolver object
#ifdef USE_DEVICE
  BLS = SUNLinSol_MagmaDense(wsubvecs[5], A, udata.ctx);
  if(check_flag((void *) BLS, "SUNLinSol_MagmaDense", 0)) return(1);
#else
  BLS = SUNLinSol_KLU(wsubvecs[5], A, udata.ctx);
  if (check_flag((void*) BLS, "SUNLinSol_KLU (main)", 0)) MPI_Abort(udata.comm, 1);
#endif

  // create linear solver wrapper and attach the matrix and linear solver to the
  // integrator and set the Jacobian for direct linear solvers
  LS = SUNLinSol_RankLocalLS(BLS, wloc, 5, &udata, inner_arkode_mem, opts, udata.ctx);
  if (check_flag((void*) LS, "SUNLinSol_RankLocalLS (main)", 0)) MPI_Abort(udata.comm, 1);

  retval = ARKStepSetLinearSolver(inner_arkode_mem, LS, A);
  if (check_flag(&retval, "ARKStepSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);

  retval = ARKStepSetJacFn(inner_arkode_mem, Jfast);
  if (check_flag(&retval, "ARKStepSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);

  // set diagnostics file
  if (udata.showstats && outproc) {
    retval = ARKStepSetDiagnostics(inner_arkode_mem, DFID_INNER);
    if (check_flag(&retval, "ARKStepSetDiagnostics (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set inner RK Butcher table
  if (opts.itable != ARKODE_DIRK_NONE) {
    retval = ARKStepSetTableNum(inner_arkode_mem, opts.itable, ARKODE_ERK_NONE);
    if (check_flag(&retval, "ARKStepSetTableNum (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set adaptive timestepping parameters (if applicable)
  if (opts.fixedstep != 1) {

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

  // otherwise, set fixed timestep size
  } else {

    retval = ARKStepSetFixedStep(inner_arkode_mem, opts.hmax);
    if (check_flag(&retval, "ARKStepSetFixedStep (main)", 1)) MPI_Abort(udata.comm, 1);

  }

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
  outer_arkode_mem = MRIStepCreate(fslow, NULL, udata.t0, w, stepper, udata.ctx);
  if (check_flag((void*) outer_arkode_mem, "MRIStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);

  retval = udata.profile[PR_MRISETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);

  // pass udata to user functions
  retval = MRIStepSetUserData(outer_arkode_mem, (void *) (&udata));
  if (check_flag(&retval, "MRIStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  // set diagnostics file
  if (udata.showstats && outproc) {
    retval = MRIStepSetDiagnostics(outer_arkode_mem, DFID_OUTER);
    if (check_flag(&retval, "MRIStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set MRI coupling table (if specified)
  if (opts.mtable != ARKODE_MRI_NONE) {
    MRIStepCoupling Gamma = MRIStepCoupling_LoadTable(opts.mtable);
    retval = MRIStepSetCoupling(outer_arkode_mem, Gamma);
    MRIStepCoupling_Free(Gamma);
    if (check_flag(&retval, "MRIStepSetCoupling (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set slow time step size
  retval = MRIStepSetFixedStep(outer_arkode_mem, hslow);
  if (check_flag(&retval, "MRIStepSetFixedStep (main)", 1)) MPI_Abort(udata.comm, 1);

  // // set routine to transfer solution information from fast to slow scale
  // retval = MRIStepSetPostInnerFn(outer_arkode_mem, PostprocessFast);
  // if (check_flag(&retval, "MRIStepSetPostInnerFn (main)", 1)) MPI_Abort(udata.comm, 1);

  retval = udata.profile[PR_MRISETUP].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

  // finish initialization
  realtype t = udata.t0;
  realtype tout = udata.t0+dTout;
  realtype hcur;

  //--- Initial batch of outputs ---//
  retval = udata.profile[PR_IO].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);

  //    Optionally output total mass/energy
  if (udata.showstats) {
    retval = check_conservation(udata.t0, w, udata);
    if (check_flag(&retval, "check_conservation (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    Output initial conditions to disk
  retval = apply_Dengo_scaling(w, udata);
  if (check_flag(&retval, "apply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
  if (NoOutput == 0) {
    retval = output_solution(udata.t0, w, opts.h0, restart, udata, opts);
    if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  //    Output CGS solution statistics (if requested)
  if (udata.showstats && PRINT_CGS == 1) {
    retval = print_stats(t, w, 0, PRINT_SCIENTIFIC, PRINT_CGS, outer_arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  retval = unapply_Dengo_scaling(w, udata);
  if (check_flag(&retval, "unapply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
  //    Output normalized solution statistics (if requested)
  if (udata.showstats && PRINT_CGS == 0) {
    retval = print_stats(t, w, 0, PRINT_SCIENTIFIC, PRINT_CGS, outer_arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  //    Output problem-specific diagnostic information
  retval = output_diagnostics(udata.t0, w, udata);
  if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

  // stop IO profiler
  retval = udata.profile[PR_IO].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

  // stop problem setup profiler
  retval = udata.profile[PR_SETUP].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

#ifndef SETUP_ONLY
  //--- Initial transient evolution: call MRIStepEvolve to perform integration   ---//
  //--- over [t0,t0+htrans], then disable adaptivity and set fixed-step size     ---//
  //--- to use for remainder of fast time scale simulation.                      ---//
  if (opts.fixedstep == 2) {

    // start transient solver profiler
    retval = udata.profile[PR_TRANS].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);

    // set transient stop time
    tout = udata.t0+opts.htrans;
    retval = MRIStepSetStopTime(outer_arkode_mem, tout);
    if (check_flag(&retval, "MRIStepSetStopTime (main)", 1)) MPI_Abort(udata.comm, 1);

    // adaptive fast timescale evolution over [t0,t0+htrans]
    retval = MRIStepEvolve(outer_arkode_mem, tout, w, &t, ARK_NORMAL);
    if (retval < 0) {    // unsuccessful solve: break
      if (outproc)  cerr << "Solver failure, stopping integration\n";
      cleanup(&outer_arkode_mem, &inner_arkode_mem, stepper, inner_content,
              udata, BLS, LS, A, w, wloc, atols, wsubvecs, Nsubvecs);
      return(1);
    }

    // stop transient solver profiler
    retval = udata.profile[PR_TRANS].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

    // output transient solver statistics
    long int nsts, nfs_e, nfs_i, nstf_max, nstf_min, nstf_a_max, nstf_a_min;
    long int nffi_max, nffi_min, netf_max, netf_min, nni_max, nni_min;
    long int ncf_max, ncf_min, nls_max, nls_min, nje_max, nje_min;
    long int nli_max, nli_min, nlcf_max, nlcf_min, nlfs_max, nlfs_min;
    nsts = nfs_e = nfs_i = 0;
    retval = MRIStepGetNumSteps(outer_arkode_mem, &nsts);
    if (check_flag(&retval, "MRIStepGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = MRIStepGetNumRhsEvals(outer_arkode_mem, &nfs_e, &nfs_i);
    if (check_flag(&retval, "MRIStepGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = RankLocalStepper_GetStats(inner_arkode_mem, udata, LS, nstf_max, nstf_min,
                                       nstf_a_max, nstf_a_min, nffi_max, nffi_min,
                                       netf_max, netf_min, nni_max, nni_min, ncf_max,
                                       ncf_min, nls_max, nls_min, nje_max, nje_min, nli_max,
                                       nli_min, nlcf_max, nlcf_min, nlfs_max, nlfs_min);
    if (check_flag(&retval, "RankLocalStepper_GetStats (main)", 1)) MPI_Abort(udata.comm, 1);
    if (outproc) {
      cout << "\nTransient portion of simulation complete:\n";
      cout << "   Slow solver steps = " << nsts << "\n";
      cout << "   Fast solver steps = (" << nstf_min << ", " << nstf_max
           << "), attempted = (" << nstf_a_min << ", " << nstf_a_max << ")\n";
      cout << "   Total RHS evals:  Fs = " << nfs_e << ",  Ff = (" << nffi_min
           << ", " << nffi_max << ")\n";
      cout << "   Total number of fast error test failures = (" << netf_min << ", "
           << netf_max << ")\n";
      if (nls_max > 0) {
        cout << "   Total number of fast lin solv setups = (" << nls_min << ", "
             << nls_max << ")\n";
        cout << "   Total number of fast Jac evals = (" << nje_min << ", "
             << nje_max << ")\n";
      }
      if (nni_max > 0) {
        cout << "   Total number of fast nonlin iters = (" << nni_min << ", "
             << nni_max << ")\n";
        cout << "   Total number of fast nonlin conv fails = (" << ncf_min << ", "
             << ncf_max << ")\n";
      }
      cout << "\nCurrent profiling results:\n";
    }
    udata.profile[PR_SETUP].print_cumulative_times("setup");
    udata.profile[PR_CHEMSETUP].print_cumulative_times("chemSetup");
    udata.profile[PR_MRISETUP].print_cumulative_times("MRIsetup");
    udata.profile[PR_IO].print_cumulative_times("I/O");
    udata.profile[PR_MPI].print_cumulative_times("MPI");
    udata.profile[PR_PACKDATA].print_cumulative_times("pack");
    udata.profile[PR_FACEFLUX].print_cumulative_times("flux");
    udata.profile[PR_RHSEULER].print_cumulative_times("Euler RHS");
    udata.profile[PR_RHSSLOW].print_cumulative_times("slow RHS");
    udata.profile[PR_RHSFAST].print_cumulative_times("fast RHS");
    udata.profile[PR_JACFAST].print_cumulative_times("fast Jac");
    udata.profile[PR_LSETUP].print_cumulative_times("lsetup");
    udata.profile[PR_LSOLVE].print_cumulative_times("lsolve");
    udata.profile[PR_MPISYNC].print_cumulative_times("MPI sync");
    // udata.profile[PR_POSTFAST].print_cumulative_times("fast post");
    udata.profile[PR_DTSTAB].print_cumulative_times("dt_stab");
    udata.profile[PR_TRANS].print_cumulative_times("trans");
    if (outproc)  cout << std::endl;

    // reset current evolution-related profilers for subsequent fixed-step evolution
    udata.profile[PR_IO].reset();
    udata.profile[PR_MPI].reset();
    udata.profile[PR_PACKDATA].reset();
    udata.profile[PR_FACEFLUX].reset();
    udata.profile[PR_RHSEULER].reset();
    udata.profile[PR_RHSSLOW].reset();
    udata.profile[PR_RHSFAST].reset();
    udata.profile[PR_JACFAST].reset();
    udata.profile[PR_LSETUP].reset();
    udata.profile[PR_LSOLVE].reset();
    udata.profile[PR_MPISYNC].reset();
    // udata.profile[PR_POSTFAST].reset();
    udata.profile[PR_DTSTAB].reset();

    // periodic output of solution/statistics
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);

    //    output diagnostic information (if applicable)
    retval = output_diagnostics(t, w, udata);
    if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

    //    output normalized statistics to stdout (if requested)
    if (udata.showstats) {
      if (PRINT_CGS == 1) {
        retval = apply_Dengo_scaling(w, udata);
        if (check_flag(&retval, "apply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
      }
      retval = print_stats(t, w, 0, PRINT_SCIENTIFIC, PRINT_CGS, outer_arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
      if (PRINT_CGS == 1) {
        retval = unapply_Dengo_scaling(w, udata);
        if (check_flag(&retval, "unapply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
      }
    }
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);


    // disable adaptivity and set fixed fast step size
    retval = ARKStepSetFixedStep(inner_arkode_mem, opts.hmax);
    if (check_flag(&retval, "ARKStepSetFixedStep (main)", 1)) MPI_Abort(udata.comm, 1);

  }


  //--- Main time-stepping loop: calls MRIStepEvolve to perform the integration, ---//
  //--- then prints results.  Stops when the final time has been reached.        ---//
  retval = udata.profile[PR_SIMUL].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);
  tout = udata.t0+dTout;
  for (int iout=restart; iout<restart+udata.nout; iout++) {

    // set stop time for next evolution
    retval = MRIStepSetStopTime(outer_arkode_mem, tout);
    if (check_flag(&retval, "MRIStepSetStopTime (main)", 1)) MPI_Abort(udata.comm, 1);

    // evolve solution
    retval = MRIStepEvolve(outer_arkode_mem, tout, w, &t, ARK_NORMAL);
    if (retval >= 0) {                         // successful solve: update output time
      tout = min(tout+dTout, udata.tf);
    } else {                                   // unsuccessful solve: break
      if (outproc)  cerr << "Solver failure, stopping integration\n";
      cleanup(&outer_arkode_mem, &inner_arkode_mem, stepper, inner_content,
              udata, BLS, LS, A, w, wloc, atols, wsubvecs, Nsubvecs);
      return(1);
    }

    // periodic output of solution/statistics
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);

    //    output diagnostic information (if applicable)
    retval = output_diagnostics(t, w, udata);
    if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

    //    output normalized statistics to stdout (if requested)
    if (udata.showstats && (PRINT_CGS == 0)) {
      retval = print_stats(t, w, 1, PRINT_SCIENTIFIC, PRINT_CGS, outer_arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    //    output results to disk -- get current step from MRIStep first
    retval = MRIStepGetLastStep(outer_arkode_mem, &hcur);
    if (check_flag(&retval, "MRIStepGetLastStep (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = apply_Dengo_scaling(w, udata);
    if (check_flag(&retval, "apply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
    if (NoOutput == 0) {
      retval = output_solution(t, w, hcur, iout+1, udata, opts);
      if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
    }
    //    output CGS statistics to stdout (if requested)
    if (udata.showstats && (PRINT_CGS == 1)) {
      retval = print_stats(t, w, 1, PRINT_SCIENTIFIC, PRINT_CGS, outer_arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }
    retval = unapply_Dengo_scaling(w, udata);
    if (check_flag(&retval, "unapply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

  }
  if (udata.showstats) {
    retval = udata.profile[PR_IO].start();
    if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = print_stats(t, w, 2, PRINT_SCIENTIFIC, PRINT_CGS, outer_arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  if (udata.showstats && outproc) {
    fclose(DFID_OUTER);
    fclose(DFID_INNER);
  }

  // compute simulation time, total time
  retval = udata.profile[PR_SIMUL].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
#endif
  retval = udata.profile[PR_TOTAL].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

  // Print some final statistics
  long int nsts, nfs_e, nfs_i, nstf_max, nstf_min, nstf_a_max, nstf_a_min;
  long int nffi_max, nffi_min, netf_max, netf_min, nni_max, nni_min;
  long int ncf_max, ncf_min, nls_max, nls_min, nje_max, nje_min;
  long int nli_max, nli_min, nlcf_max, nlcf_min, nlfs_max, nlfs_min;
  nsts = nfs_e = nfs_i = 0;
  retval = MRIStepGetNumSteps(outer_arkode_mem, &nsts);
  if (check_flag(&retval, "MRIStepGetNumSteps (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = MRIStepGetNumRhsEvals(outer_arkode_mem, &nfs_e, &nfs_i);
  if (check_flag(&retval, "MRIStepGetNumRhsEvals (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = RankLocalStepper_GetStats(inner_arkode_mem, udata, LS, nstf_max, nstf_min,
                                     nstf_a_max, nstf_a_min, nffi_max, nffi_min,
                                     netf_max, netf_min, nni_max, nni_min, ncf_max,
                                     ncf_min, nls_max, nls_min, nje_max, nje_min, nli_max,
                                     nli_min, nlcf_max, nlcf_min, nlfs_max, nlfs_min);
  if (check_flag(&retval, "RankLocalStepper_GetStats (main)", 1)) MPI_Abort(udata.comm, 1);
  if (outproc) {
    cout << "\nOverall Solver Statistics:\n";
    cout << "   Slow solver steps = " << nsts << "\n";
    cout << "   Fast solver steps = (" << nstf_min << ", " << nstf_max
         << "), attempted = (" << nstf_a_min << ", " << nstf_a_max << ")\n";
    cout << "   Total RHS evals:  Fs = " << nfs_e << ",  Ff = (" << nffi_min
         << ", " << nffi_max << ")\n";
    cout << "   Total number of fast error test failures = (" << netf_min << ", "
         << netf_max << ")\n";
    if (nls_max > 0) {
      cout << "   Total number of fast lin solv setups = (" << nls_min << ", "
           << nls_max << ")\n";
      cout << "   Total number of fast Jac evals = (" << nje_min << ", "
           << nje_max << ")\n";
    }
    if (nni_max > 0) {
      cout << "   Total number of fast nonlin iters = (" << nni_min << ", "
           << nni_max << ")\n";
      cout << "   Total number of fast nonlin conv fails = (" << ncf_min << ", "
           << ncf_max << ")\n";
    }
    cout << "\nFinal profiling results:\n";
  }
  udata.profile[PR_SETUP].print_cumulative_times("setup");
  udata.profile[PR_CHEMSETUP].print_cumulative_times("chemSetup");
  udata.profile[PR_MRISETUP].print_cumulative_times("MRIsetup");
  udata.profile[PR_IO].print_cumulative_times("I/O");
  udata.profile[PR_MPI].print_cumulative_times("MPI");
  udata.profile[PR_PACKDATA].print_cumulative_times("pack");
  udata.profile[PR_FACEFLUX].print_cumulative_times("flux");
  udata.profile[PR_RHSEULER].print_cumulative_times("Euler RHS");
  udata.profile[PR_RHSSLOW].print_cumulative_times("slow RHS");
  udata.profile[PR_RHSFAST].print_cumulative_times("fast RHS");
  udata.profile[PR_JACFAST].print_cumulative_times("fast Jac");
  udata.profile[PR_LSETUP].print_cumulative_times("lsetup");
  udata.profile[PR_LSOLVE].print_cumulative_times("lsolve");
  udata.profile[PR_MPISYNC].print_cumulative_times("MPI sync");
  // udata.profile[PR_POSTFAST].print_cumulative_times("fast post");
  udata.profile[PR_DTSTAB].print_cumulative_times("dt_stab");
  udata.profile[PR_SIMUL].print_cumulative_times("sim");
  udata.profile[PR_TOTAL].print_cumulative_times("Total");

  // Output mass/energy conservation error
  if (udata.showstats) {
    if (outproc)  cout << "\nConservation Check:\n";
    retval = check_conservation(t, w, udata);
    if (check_flag(&retval, "check_conservation (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // Clean up, finalize MPI, and return with successful completion
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  cleanup(&outer_arkode_mem, &inner_arkode_mem, stepper, inner_content,
          udata, BLS, LS, A, w, wloc, atols, wsubvecs, Nsubvecs);
  udata.FreeData();
  MPI_Finalize();                  // Finalize MPI
  return 0;
}


//---- fast/slow problem-defining functions (wrappers for other routines) ----

static int ffast(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // access problem data structure and start timer
  RankLocalStepperContent *inner_content = (RankLocalStepperContent*) user_data;
  EulerData *udata = inner_content->udata;
  int retval = udata->profile[PR_RHSFAST].start();
  if (check_flag(&retval, "Profile::start (ffast)", 1)) return(-1);

  // initialize all outputs to zero
  N_VConst(ZERO, wdot);

  // call Dengo RHS routine on chemistry subvectors
  N_Vector wchem = NULL;
  wchem = N_VGetSubvector_ManyVector(w, 5);
  if (check_flag((void *) wchem, "N_VGetSubvector_ManyVector (ffast)", 0))  return(-1);
  N_Vector wchemdot = NULL;
  wchemdot = N_VGetSubvector_ManyVector(wdot, 5);
  if (check_flag((void *) wchemdot, "N_VGetSubvector_ManyVector (ffast)", 0))  return(-1);
  retval = calculate_rhs_cvklu(t, wchem, wchemdot,
                               (udata->nxl)*(udata->nyl)*(udata->nzl),
                               udata->RxNetData);
  if (check_flag(&retval, "calculate_rhs_cvklu (ffast)", 1)) return(retval);

  // scale wdot by TimeUnits to handle step size nondimensionalization
  N_VScale(udata->TimeUnits, wchemdot, wchemdot);

  // update wdot with forcing terms from slow time scale (if applicable)
  if (!inner_content->disable_forcing) {

    int nforcing;
    realtype tshift, tscale;
    N_Vector *forcing;   // Note: this is an array of MPIManyVectors
    retval = MRIStepInnerStepper_GetForcingData(inner_content->stepper, &tshift,
                                                &tscale, &forcing, &nforcing);
    if (check_flag(&retval, "MRIStepInnerStepper_GetForcingData (ffast)", 1)) return(retval);

    // apply forcing separately on each component vector
    N_Vector Xvecs[10];   // to be safe, set arrays much longer than needed
    realtype cvals[10] = {ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE};
    realtype tau = (t-tshift) / tscale;
    for (int ivec=0; ivec<6; ivec++) {
      Xvecs[0] = N_VGetSubvector_ManyVector(wdot, ivec);
      if (check_flag((void *) Xvecs[0], "N_VGetSubvector_ManyVector (ffast)", 0))  return(-1);
      for (int i=0; i<nforcing; i++) {
        cvals[i+1] = SUNRpowerI(tau, i);
        Xvecs[i+1] = N_VGetSubvector_MPIManyVector(forcing[i], ivec);
        if (check_flag((void *) Xvecs[i+1], "N_VGetSubvector_MPIManyVector (ffast)", 0))  return(-1);
      }
      retval = N_VLinearCombination(nforcing+1, cvals, Xvecs, Xvecs[0]);
      if (check_flag(&retval, "N_VLinearCombination (ffast)", 1)) return(retval);
    }
  }

  // stop timer and return
  retval = udata->profile[PR_RHSFAST].stop();
  if (check_flag(&retval, "Profile::stop (ffast)", 1)) return(-1);
  return(0);
}

static int Jfast(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                 void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // access problem data structure and start timer
  RankLocalStepperContent *inner_content = (RankLocalStepperContent*) user_data;
  EulerData *udata = inner_content->udata;
  int retval = udata->profile[PR_JACFAST].start();
  if (check_flag(&retval, "Profile::start (Jfast)", 1)) return(-1);

  // call Jacobian routine
  N_Vector wchem = N_VGetSubvector_ManyVector(w, 5);
  if (check_flag((void *) wchem, "N_VGetSubvector_ManyVector (Jfast)", 0))  return(-1);
  N_Vector fwchem = N_VGetSubvector_ManyVector(fw, 5);
  if (check_flag((void *) fwchem, "N_VGetSubvector_ManyVector (Jfast)", 0))  return(-1);
  retval = calculate_jacobian_cvklu(t, wchem, fwchem, Jac,
                                    (udata->nxl)*(udata->nyl)*(udata->nzl),
                                    udata->RxNetData, tmp1, tmp2, tmp3);
  if (check_flag(&retval, "calculate_jacobian_cvklu (Jfast)", 1)) return(retval);

  // scale Jac values by TimeUnits to handle step size nondimensionalization
  realtype *Jdata = NULL;
  realtype TUnit = udata->TimeUnits;
#ifdef USE_DEVICE
  Jdata = SUNMatrix_MagmaDense_Data(Jac);
  if (check_flag((void *) Jdata, "SUNMatrix_MagmaDense_Data (Jimpl)", 0)) return(-1);
  long int ldata = SUNMatrix_MagmaDense_LData(Jac);
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,ldata), [=] RAJA_DEVICE (long int i) {
    Jdata[i] *= TUnit;
  });
#else
  Jdata = SUNSparseMatrix_Data(Jac);
  if (check_flag((void *) Jdata, "SUNSparseMatrix_Data (Jimpl)", 0)) return(-1);
  sunindextype nnz = SUNSparseMatrix_NNZ(Jac);
  for (sunindextype i=0; i<nnz; i++)  Jdata[i] *= TUnit;
#endif

  // stop timer and return
  retval = udata->profile[PR_JACFAST].stop();
  if (check_flag(&retval, "Profile::stop (Jfast)", 1)) return(-1);
  return(0);
}

static int fslow(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // start timer
  EulerData *udata = (EulerData*) user_data;
  int retval = udata->profile[PR_RHSSLOW].start();
  if (check_flag(&retval, "Profile::start (fslow)", 1)) return(-1);

  // initialize all outputs to zero (necessary??)
  N_VConst(ZERO, wdot);

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);
  realtype *etdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,4);
  if (check_flag((void *) etdot, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);
  realtype *chemdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,5);
  if (check_flag((void *) chemdot, "N_VGetSubvectorArrayPointer (fslow)", 0)) return(-1);

  // update chem to include Dengo scaling
  retval = apply_Dengo_scaling(w, *udata);
  if (check_flag(&retval, "apply_Dengo_scaling (fslow)", 1)) return(-1);

#ifdef USE_DEVICE
  // ensure that chemistry data is synchronized to host
  N_VCopyFromDevice_Raja(N_VGetSubvector_MPIManyVector(w,5));
#endif

  // fill dimensionless total fluid energy field (internal energy + kinetic energy)
  realtype EUnitScale = ONE/udata->EnergyUnits;
  for (int k=0; k<udata->nzl; k++)
    for (int j=0; j<udata->nyl; j++)
      for (int i=0; i<udata->nxl; i++) {
        const long int cidx = BUFINDX(udata->nchem-1,i,j,k,udata->nchem,udata->nxl,udata->nyl,udata->nzl);
        const long int fidx = INDX(i,j,k,udata->nxl,udata->nyl,udata->nzl);
        et[fidx] = chem[cidx] * EUnitScale
          + 0.5/rho[fidx]*(mx[fidx]*mx[fidx] + my[fidx]*my[fidx] + mz[fidx]*mz[fidx]);
      }

#ifndef DISABLE_HYDRO
  // call fEuler as usual
  retval = fEuler(t, w, wdot, user_data);
  if (check_flag(&retval, "fEuler (fslow)", 1)) return(retval);
#endif

  // overwrite chemistry energy "fslow" with total energy "fslow" (with
  // appropriate unit scaling) and zero out total energy fslow
  //
  // QUESTION: is this really necessary, since fEuler also advects chemistry gas energy?
  // PARTIAL ANSWER: the external forces are currently only applied to the fluid fields,
  //   so these need to additionally force the chemistry gas energy
  //
  // Note: fEuler computes dy/dtau where tau = t / TimeUnits, but chemistry
  // RHS should compute dy/dt = dy/dtau * dtau/dt = dy/dtau * 1/TimeUnits
//  realtype TUnitScale = ONE/udata->TimeUnits;
  realtype TUnitScale = ONE;
  for (int k=0; k<udata->nzl; k++)
    for (int j=0; j<udata->nyl; j++)
      for (int i=0; i<udata->nxl; i++) {
        const long int cidx = BUFINDX(udata->nchem-1,i,j,k,udata->nchem,udata->nxl,udata->nyl,udata->nzl);
        const long int fidx = INDX(i,j,k,udata->nxl,udata->nyl,udata->nzl);
        chemdot[cidx] = etdot[fidx]*TUnitScale;
        etdot[fidx] = ZERO;
      }

#ifdef USE_DEVICE
  // ensure that chemistry rate-of-change data is synchronized back to device
  N_VCopyToDevice_Raja(N_VGetSubvector_MPIManyVector(wdot,5));
#endif

  // reset chem to remove Dengo scaling
  retval = unapply_Dengo_scaling(w, *udata);
  if (check_flag(&retval, "unapply_Dengo_scaling (fslow)", 1)) return(-1);

  // stop timer and return
  retval = udata->profile[PR_RHSSLOW].stop();
  if (check_flag(&retval, "Profile::stop (fslow)", 1)) return(-1);
  return(0);
}


//---- utility routines ----

void cleanup(void **outer_arkode_mem, void **inner_arkode_mem,
             MRIStepInnerStepper stepper,
             RankLocalStepperContent *inner_content, EulerData& udata,
             SUNLinearSolver BLS, SUNLinearSolver LS, SUNMatrix A, N_Vector w,
             N_Vector wloc, N_Vector atols, N_Vector *wsubvecs, int Nsubvecs)
{
  delete inner_content;
  MRIStepInnerStepper_Free(&stepper);
  MRIStepFree(outer_arkode_mem);   // Free integrator memory
  ARKStepFree(inner_arkode_mem);
  SUNLinSolFree(BLS);              // Free matrix and linear solvers
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  for (int i=0; i<Nsubvecs; i++)   // Free solution/tolerance vectors
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  N_VDestroy(w);
  N_VDestroy(wloc);
  N_VDestroy(atols);
  free_Dengo_structures(udata);
}


//---- custom MRIStepInnerStepper module ----

//   Rank-local stepper time evolution routine
static int RankLocalStepper_Evolve(MRIStepInnerStepper stepper, realtype t0,
                                   realtype tout, N_Vector y) {

  // access inner stepper content structure
  int retval;
  void *inner_content = NULL;
  retval = MRIStepInnerStepper_GetContent(stepper, &inner_content);
  if (check_flag(&retval, "MRIStepInnerStepper_GetContent (RankLocalStepper_Evolve)", 1))
    MPI_Abort(MPI_COMM_WORLD, 1);
  RankLocalStepperContent *content = (RankLocalStepperContent*) inner_content;

  // create ManyVector version of input MPIManyVector (reuse y's context object)
  N_Vector ysubvecs[6];
  for (int ivec=0; ivec<6; ivec++)
    ysubvecs[ivec] = N_VGetSubvector_MPIManyVector(y, ivec);
  N_Vector yloc = N_VNew_ManyVector(6, ysubvecs, y->sunctx);
  if (check_flag((void *) yloc, "N_VNewManyVector (RankLocalStepper_Evolve)", 0))
    return(-1);

  // set the stop time for the rank-local ARKStep solver
  retval = ARKStepSetStopTime(content->solver_mem, tout);
  if (check_flag(&retval, "ARKStepSetStopTime (RankLocalStepper_Evolve)", 1)) {
    N_VDestroy(yloc);
    return(1);
  }

  // call ARKStepEvolve to perform fast integration
  realtype tret;
  retval = ARKStepEvolve(content->solver_mem, tout, yloc, &tret, ARK_NORMAL);

  // free ManyVector wrapper
  N_VDestroy(yloc);

  // determine return flag via reduction across ranks
  int ierrs[2], globerrs[2];
  ierrs[0] = retval; ierrs[1] = -retval;
  retval = content->udata->profile[PR_MPISYNC].start();
  if (check_flag(&retval, "Profile::start (RankLocalStepper_Evolve)", 1))  return(-1);
  retval = MPI_Allreduce(ierrs, globerrs, 2, MPI_INT, MPI_MIN, content->udata->comm);
  if (check_flag(&retval, "MPI_Alleduce (RankLocalStepper_Evolve)", 3)) return(-1);
  retval = content->udata->profile[PR_MPISYNC].stop();
  if (check_flag(&retval, "Profile::stop (RankLocalStepper_Evolve)", 1))  return(-1);

  // return unrecoverable failure if relevant;
  // otherwise return the success and/or recoverable failure flag
  if (globerrs[0] < 0) return(globerrs[0]);
  else                 return(-globerrs[1]);
}


//   Rank-local stepper full right-hand side calculation routine
static int RankLocalStepper_FullRhs(MRIStepInnerStepper stepper, realtype t,
                                    N_Vector y, N_Vector f, int mode) {

  // access inner stepper content structure
  int retval;
  void *inner_content = NULL;
  retval = MRIStepInnerStepper_GetContent(stepper, &inner_content);
  if (check_flag(&retval, "MRIStepInnerStepper_GetContent (RankLocalStepper_FullRhs)", 1))
    MPI_Abort(MPI_COMM_WORLD, 1);
  RankLocalStepperContent *content = (RankLocalStepperContent*) inner_content;

  // create ManyVector versions of input MPIManyVectors (reuse context objects)
  N_Vector subvecs[6];
  for (int ivec=0; ivec<6; ivec++)
    subvecs[ivec] = N_VGetSubvector_MPIManyVector(y, ivec);
  N_Vector yloc = N_VNew_ManyVector(6, subvecs, y->sunctx);
  if (check_flag((void *) yloc, "N_VNewManyVector (RankLocalStepper_FullRhs)", 0))
    return(-1);
  for (int ivec=0; ivec<6; ivec++)
    subvecs[ivec] = N_VGetSubvector_MPIManyVector(f, ivec);
  N_Vector floc = N_VNew_ManyVector(6, subvecs, f->sunctx);
  if (check_flag((void *) floc, "N_VNewManyVector (RankLocalStepper_FullRhs)", 0))
    return(-1);

  // call ffast with forcing disabled
  content->disable_forcing = true;
  retval = ffast(t, yloc, floc, inner_content);
  content->disable_forcing = false;

  // free ManyVector wrappers and return
  N_VDestroy(yloc);
  N_VDestroy(floc);
  return retval;
}


//   Rank-local stepper solver "reset" routine
static int RankLocalStepper_Reset(MRIStepInnerStepper stepper, realtype tR, N_Vector yR) {

  // access inner stepper content structure
  int retval;
  void *inner_content = NULL;
  retval = MRIStepInnerStepper_GetContent(stepper, &inner_content);
  if (check_flag(&retval, "MRIStepInnerStepper_GetContent (RankLocalStepper_Reset)", 1))
    MPI_Abort(MPI_COMM_WORLD, 1);
  RankLocalStepperContent *content = (RankLocalStepperContent*) inner_content;

  // create ManyVector versions of input MPIManyVectors (reuse context objects)
  N_Vector ysubvecs[6];
  for (int ivec=0; ivec<6; ivec++)
    ysubvecs[ivec] = N_VGetSubvector_MPIManyVector(yR, ivec);
  N_Vector yloc = N_VNew_ManyVector(6, ysubvecs, yR->sunctx);
  if (check_flag((void *) yloc, "N_VNewManyVector (RankLocalStepper_Reset)", 0))
    return(-1);

  // call ARKStep reset routine, free ManyVector wrapper, and return
  retval = ARKStepReset(content->solver_mem, tR, yloc);
  N_VDestroy(yloc);
  return retval;
}


//   Rank-local stepper statistics retrieval (must be called on all MPI ranks)
static int RankLocalStepper_GetStats(void* arkode_mem, EulerData& udata,
                                     SUNLinearSolver& ranklocalLS,
                                     long int& nst_max, long int& nst_min,
                                     long int& nst_a_max, long int& nst_a_min,
                                     long int& nfi_max, long int& nfi_min,
                                     long int& netf_max, long int& netf_min,
                                     long int& nni_max, long int& nni_min,
                                     long int& ncf_max, long int& ncf_min,
                                     long int& nls_max, long int& nls_min,
                                     long int& nje_max, long int& nje_min,
                                     long int& nli_max, long int& nli_min,
                                     long int& nlcf_max, long int& nlcf_min,
                                     long int& nlfs_max, long int& nlfs_min) {

  // access statistics from rank-local integrator
  int retval;
  long int nst, nst_a, nfe, nfi, netf, nni, ncf, nls, nje, nli, nlcf, nlfs;
  nst = nst_a = nfe = nfi = netf = nni = ncf = nls = nje = nli = nlcf = nlfs = 0;
  retval = ARKStepGetNumSteps(arkode_mem, &nst);
  if (check_flag(&retval, "ARKStepGetNumSteps (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumStepAttempts(arkode_mem, &nst_a);
  if (check_flag(&retval, "ARKStepGetNumStepAttempts (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumRhsEvals(arkode_mem, &nfe, &nfi);
  if (check_flag(&retval, "ARKStepGetNumRhsEvals (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumErrTestFails(arkode_mem, &netf);
  if (check_flag(&retval, "ARKStepGetNumErrTestFails (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNonlinSolvStats(arkode_mem, &nni, &ncf);
  if (check_flag(&retval, "ARKStepGetNonlinSolvStats (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinSolvSetups(arkode_mem, &nls);
  if (check_flag(&retval, "ARKStepGetNumLinSolvSetups (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumJacEvals(arkode_mem, &nje);
  if (check_flag(&retval, "ARKStepGetNumJacEvals (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinIters(arkode_mem, &nli);
  if (check_flag(&retval, "ARKStepGetNumLinIters (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  retval = ARKStepGetNumLinConvFails(arkode_mem, &nlcf);
  if (check_flag(&retval, "ARKStepGetNumLinConvFails (RankLocalStepper_GetStats)", 1))
    MPI_Abort(udata.comm, 1);
  nlfs = RankLocalLS_NFEDQ(ranklocalLS);

  // Perform MPI reductions to determine min/max for each statistic
  long int stats[11] = {nst, nst_a, nfi, netf, nni, ncf, nls, nje, nli, nlcf, nlfs};
  long int stats_min[11];
  long int stats_max[11];
  retval = MPI_Reduce(stats, stats_min, 11, MPI_LONG, MPI_MIN, 0, udata.comm);
  if (retval != MPI_SUCCESS)  return 1;
  retval = MPI_Reduce(stats, stats_max, 11, MPI_LONG, MPI_MAX, 0, udata.comm);
  if (retval != MPI_SUCCESS)  return 1;

  // unpack statistics and return
  nst_max   = stats_max[0];
  nst_a_max = stats_max[1];
  nfi_max   = stats_max[2];
  netf_max  = stats_max[3];
  nni_max   = stats_max[4];
  ncf_max   = stats_max[5];
  nls_max   = stats_max[6];
  nje_max   = stats_max[7];
  nli_max   = stats_max[8];
  nlcf_max  = stats_max[9];
  nlfs_max  = stats_max[10];
  nst_min   = stats_min[0];
  nst_a_min = stats_min[1];
  nfi_min   = stats_min[2];
  netf_min  = stats_min[3];
  nni_min   = stats_min[4];
  ncf_min   = stats_min[5];
  nls_min   = stats_min[6];
  nje_min   = stats_min[7];
  nli_min   = stats_min[8];
  nlcf_min  = stats_min[9];
  nlfs_min  = stats_min[10];
  return 0;
}

//---- custom SUNLinearSolver module ----

SUNLinearSolver SUNLinSol_RankLocalLS(SUNLinearSolver BLS, N_Vector x,
                                      sunindextype subvec, EulerData* udata,
                                      void* arkode_mem,
                                      ARKODEParameters& opts, SUNContext ctx)
{
  // Check compatibility with supplied N_Vector
  if (N_VGetVectorID(x) != SUNDIALS_NVEC_MANYVECTOR) return(NULL);
  if (subvec >= N_VGetNumSubvectors_ManyVector(x)) return(NULL);

  // Create an empty linear solver
  SUNLinearSolver S = SUNLinSolNewEmpty(ctx);
  if (S == NULL) return(NULL);

  // Attach operations (use defaults whenever possible)
  S->ops->gettype     = GetType_RankLocalLS;
  S->ops->initialize  = Initialize_RankLocalLS;
  S->ops->setup       = Setup_RankLocalLS;
  S->ops->solve       = Solve_RankLocalLS;
  S->ops->lastflag    = LastFlag_RankLocalLS;
  S->ops->free        = Free_RankLocalLS;

  // Create, fill and attach content
  RankLocalLSContent content = NULL;
  content = (RankLocalLSContent) malloc(sizeof *content);
  if (content == NULL) { SUNLinSolFree(S); return(NULL); }

  content->blockLS    = BLS;
  content->subvec     = subvec;
  content->udata      = udata;
  content->arkode_mem = arkode_mem;
  content->nfeDQ      = 0;
  S->content          = content;

  return(S);
}

SUNLinearSolver_Type GetType_RankLocalLS(SUNLinearSolver S)
{
  return(SUNLINEARSOLVER_DIRECT);
}

int Initialize_RankLocalLS(SUNLinearSolver S)
{
  // pass initialize call down to block linear solver
  RankLocalLS_LASTFLAG(S) = SUNLinSolInitialize(RankLocalLS_BLS(S));
  return(RankLocalLS_LASTFLAG(S));
}

int Setup_RankLocalLS(SUNLinearSolver S, SUNMatrix A)
{
  // pass setup call down to block linear solver
  int retval = RankLocalLS_UDATA(S)->profile[PR_LSETUP].start();
  if (check_flag(&retval, "Profile::start (Setup_RankLocalLS)", 1))  return(retval);
  RankLocalLS_LASTFLAG(S) = SUNLinSolSetup(RankLocalLS_BLS(S), A);
  retval = RankLocalLS_UDATA(S)->profile[PR_LSETUP].stop();
  if (check_flag(&retval, "Profile::stop (Setup_RankLocalLS)", 1))  return(retval);
  return(RankLocalLS_LASTFLAG(S));
}

int Solve_RankLocalLS(SUNLinearSolver S, SUNMatrix A,
                      N_Vector x, N_Vector b, realtype tol)
{
  // start profiling timer
  int retval = RankLocalLS_UDATA(S)->profile[PR_LSOLVE].start();
  if (check_flag(&retval, "Profile::start (Solve_RankLocalLS)", 1))  return(-1);

  // access desired subvector from ManyVector objects
  N_Vector xsub = N_VGetSubvector_ManyVector(x, RankLocalLS_SUBVEC(S));
  N_Vector bsub = N_VGetSubvector_ManyVector(b, RankLocalLS_SUBVEC(S));
  if ((xsub == NULL) || (bsub == NULL)) {
    RankLocalLS_LASTFLAG(S) = SUNLS_MEM_FAIL;
    return(RankLocalLS_LASTFLAG(S));
  }

  // pass solve call down to the block linear solver
  RankLocalLS_LASTFLAG(S) = SUNLinSolSolve(RankLocalLS_BLS(S), A, xsub, bsub, tol);

  // stop profiling timer
  retval = RankLocalLS_UDATA(S)->profile[PR_LSOLVE].stop();
  if (check_flag(&retval, "Profile::stop (Solve_RankLocalLS)", 1))  return(-1);

  // return flag from block linear solver
  return(RankLocalLS_LASTFLAG(S));
}

sunindextype LastFlag_RankLocalLS(SUNLinearSolver S)
{
  return(RankLocalLS_LASTFLAG(S));
}


int Free_RankLocalLS(SUNLinearSolver S)
{
  RankLocalLSContent content = RankLocalLS_CONTENT(S);
  if (content == NULL) return(0);
  free(S->ops);
  free(S->content);
  free(S);
  return(0);
}

//---- end of file ----
