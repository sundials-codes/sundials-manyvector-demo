/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 IMEX chemistry + hydrodynamics driver:

 The explicit portion of the RHS evolves the 3D compressible,
 inviscid Euler equations.  The implicit portion of the RHS
 evolves a chemical network provided by Dengo -- a flexible
 Python library that creates ODE RHS and Jacobian routines for
 arbitrarily-complex chemistry networks.

 The problem is evolved using ARKode's ARKStep time-stepping
 module, and is currently hard-coded to use the 4th-order
 ARK437L2SA_DIRK_7_3_4 + ARK437L2SA_ERK_7_3_4 Butcher table pair
 for a temporally adaptive additive Runge--Kutta solve.  Aside
 from this selection of Butcher tables, nearly all adaptivity
 and implicit solver options are controllable via user inputs.
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
#include <euler3D.hpp>
#ifdef USERAJA
#include <raja_primordial_network.hpp>
#else
#include <dengo_primordial_network.hpp>
#endif
#include <arkode/arkode_arkstep.h>
#ifdef RAJA_CUDA
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#else
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#endif
#include <sunlinsol/sunlinsol_spgmr.h>

#ifdef DEBUG
#include "fenv.h"
#endif

//#define DISABLE_HYDRO

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
static int fimpl(realtype t, N_Vector w, N_Vector wdot, void* user_data);
static int Jimpl(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                 void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int fexpl(realtype t, N_Vector w, N_Vector wdot, void* user_data);
static int PostprocessStep(realtype t, N_Vector y, void* user_data);

// utility routines
void cleanup(void **arkode_mem, EulerData& udata,
             SUNLinearSolver BLS, SUNLinearSolver LS, SUNMatrix A,
             N_Vector w, N_Vector atols, N_Vector *wsubvecs, int Nsubvecs);


// custom Block-Diagonal MPIManyVector SUNLinearSolver module
typedef struct _BDMPIMVContent {
  SUNLinearSolver blockLS;
  sunindextype    subvec;
  sunindextype    lastflag;
  EulerData*      udata;
  void*           arkode_mem;
  N_Vector        work;
  long int        nfeDQ;
} *BDMPIMVContent;
#define BDMPIMV_CONTENT(S)  ( (BDMPIMVContent)(S->content) )
#define BDMPIMV_BLS(S)      ( BDMPIMV_CONTENT(S)->blockLS )
#define BDMPIMV_SUBVEC(S)   ( BDMPIMV_CONTENT(S)->subvec )
#define BDMPIMV_LASTFLAG(S) ( BDMPIMV_CONTENT(S)->lastflag )
#define BDMPIMV_UDATA(S)    ( BDMPIMV_CONTENT(S)->udata )
#define BDMPIMV_WORK(S)     ( BDMPIMV_CONTENT(S)->work )
#define BDMPIMV_NFEDQ(S)    ( BDMPIMV_CONTENT(S)->nfeDQ )
SUNLinearSolver SUNLinSol_BDMPIMV(SUNLinearSolver LS, N_Vector x,
                                  sunindextype subvec, EulerData* udata,
                                  void *arkode_mem, ARKodeParameters& opts);
SUNLinearSolver_Type SUNLinSolGetType_BDMPIMV(SUNLinearSolver S);
int SUNLinSolInitialize_BDMPIMV(SUNLinearSolver S);
int SUNLinSolSetup_BDMPIMV(SUNLinearSolver S, SUNMatrix A);
int SUNLinSolSolve_BDMPIMV(SUNLinearSolver S, SUNMatrix A,
                           N_Vector x, N_Vector b, realtype tol);
int SUNLinSolSetATimes_BDMPIMV(SUNLinearSolver S, void* A_data,
                               ATimesFn ATimes);
int ATimes_BDMPIMV(void* arkode_mem, N_Vector v, N_Vector z);
sunindextype SUNLinSolLastFlag_BDMPIMV(SUNLinearSolver S);
int SUNLinSolFree_BDMPIMV(SUNLinearSolver S);


// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem variables
  long int N, Ntot;
  int Nsubvecs;
  int retval;                    // reusable error-checking flag
  int idense;                    // flag denoting integration type (dense output vs tstop)
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use (disabled if negative)
  int NoOutput;                  // flag for case when no output is desired
  N_Vector w = NULL;             // empty vectors for storing overall solution
  N_Vector *wsubvecs;
  N_Vector atols = NULL;
  SUNMatrix A = NULL;            // empty matrix and linear solver structures
  SUNLinearSolver LS = NULL;
  SUNLinearSolver BLS = NULL;
  void *arkode_mem = NULL;       // empty ARKStep memory structure
  EulerData udata;               // solver data structures
  ARKodeParameters opts;
#ifdef RAJA_CUDA
  cusparseHandle_t cusp_handle;
  cusolverSpHandle_t cusol_handle;
#endif

  //--- General Initialization ---//

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return(1);
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // start various code profilers
  retval = udata.profile[PR_TOTAL].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  retval = udata.profile[PR_SETUP].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  retval = udata.profile[PR_IO].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  if (myid == 0)  cout << "Initializing problem\n";
  retval = MPI_Barrier(MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // read problem and solver parameters from input file / command line
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  if (myid == 0)  cout << "Setting up parallel decomposition\n";
  retval = MPI_Barrier(MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

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
    cout << "\n3D compressible inviscid Euler + primordial chemistry driver (imex):\n";
    cout << "   nprocs: " << udata.nprocs << " (" << udata.npx << " x "
         << udata.npy << " x " << udata.npz << ")\n";
    cout << "   spatial domain: [" << udata.xl << ", " << udata.xr << "] x ["
         << udata.yl << ", " << udata.yr << "] x ["
         << udata.zl << ", " << udata.zr << "]\n";
    cout << "   time domain = (" << udata.t0 << ", " << udata.tf << "]"
         << ",  or (" << udata.t0*udata.TimeUnits
         << ", " << udata.tf*udata.TimeUnits << "] in CGS\n";
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
    cout << "   cfl fraction: " << udata.cfl << "\n";
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
#ifdef USERAJA
#ifdef RAJA_CUDA
    cout << "Executable built with RAJA+CUDA support\n";
#elif RAJA_SERIAL
    cout << "Executable built with RAJA+SERIAL support\n";
#else
    cout << "Executable built with RAJA+HIP support\n";
#endif
#else
    cout << "Executable built without RAJA support\n";
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
  FILE *DFID = NULL;
  if (udata.showstats && outproc) {
    DFID=fopen("diags_chem_hydro.txt","w");
  }
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);

  // Initialize N_Vector data structures with configured vector operations
  N = (udata.nxl)*(udata.nyl)*(udata.nzl);
  Ntot = (udata.nx)*(udata.ny)*(udata.nz);
  Nsubvecs = 5 + ((udata.nchem > 0) ? 1 : 0);
  wsubvecs = new N_Vector[Nsubvecs];
  for (int i=0; i<5; i++) {
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
  atols = N_VClone(w);                           // absolute tolerance vector
  if (check_flag((void *) atols, "N_VClone (main)", 0)) MPI_Abort(udata.comm, 1);
  N_VConst(opts.atol, atols);

  // initialize Dengo data structure, "network_data" (stored within udata)
  retval = initialize_Dengo_structures(udata);
  if (check_flag(&retval, "initialize_Dengo_structures (main)", 1)) MPI_Abort(udata.comm, 1);

  // set initial conditions (or restart from file)
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

  // prepare Dengo structures and initial condition vector
  retval = prepare_Dengo_structures(udata.t0, w, udata);
  if (check_flag(&retval, "prepare_Dengo_structures (main)", 1)) MPI_Abort(udata.comm, 1);


  //--- create the ARKStep integrator and set options ---//

  // initialize the integrator
  arkode_mem = ARKStepCreate(fexpl, fimpl, udata.t0, w);
  if (check_flag((void*) arkode_mem, "ARKStepCreate (main)", 0)) MPI_Abort(udata.comm, 1);

  // pass udata to user functions
  retval = ARKStepSetUserData(arkode_mem, (void *) (&udata));
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);

  // create the fast integrator local linear solver
  if (opts.iterative) {
    BLS = SUNLinSol_SPGMR(wsubvecs[5], PREC_NONE, opts.maxliters);
    if(check_flag((void*) BLS, "SUNLinSol_SPGMR (main)", 0)) MPI_Abort(udata.comm, 1);
  } else {
#ifdef RAJA_CUDA
    // Initialize cuSOLVER and cuSPARSE handles
    cusparseCreate(&cusp_handle);
    cusolverSpCreate(&cusol_handle);
    A = SUNMatrix_cuSparse_NewBlockCSR(N, udata.nchem, udata.nchem, 64*udata.nchem, cusp_handle);
    if(check_flag((void*) A, "SUNMatrix_cuSparse_NewBlockCSR (main)", 0)) MPI_Abort(udata.comm, 1);
    BLS = SUNLinSol_cuSolverSp_batchQR(wsubvecs[5], A, cusol_handle);
    if(check_flag((void*) BLS, "SUNLinSol_cuSolverSp_batchQR (main)", 0)) MPI_Abort(udata.comm, 1);
    // Set the sparsity pattern to be fixed
    retval = SUNMatrix_cuSparse_SetFixedPattern(A, SUNTRUE);
    if(check_flag(&retval, "SUNMatrix_cuSolverSp_SetFixedPattern (main)", 0)) MPI_Abort(udata.comm, 1);
    // Initialiize the Jacobian with the fixed sparsity pattern
    retval = initialize_sparse_jacobian_cvklu(A, &udata);
    if(check_flag(&retval, "initialize_sparse_jacobian_cvklu (main)", 0)) MPI_Abort(udata.comm, 1);
#else
    A = SUNSparseMatrix(N*udata.nchem, N*udata.nchem, 64*N*udata.nchem, CSR_MAT);
    if (check_flag((void*) A, "SUNSparseMatrix (main)", 0)) MPI_Abort(udata.comm, 1);
    BLS = SUNLinSol_KLU(wsubvecs[5], A);
    if (check_flag((void*) BLS, "SUNLinSol_KLU (main)", 0)) MPI_Abort(udata.comm, 1);
#endif
  }

  // create linear solver wrapper and attach the matrix and linear solver to the
  // integrator and set the Jacobian for direct linear solvers
  LS = SUNLinSol_BDMPIMV(BLS, w, 5, &udata, arkode_mem, opts);
  if (check_flag((void*) LS, "SUNLinSol_BDMPIMV (main)", 0)) MPI_Abort(udata.comm, 1);
  retval = ARKStepSetLinearSolver(arkode_mem, LS, A);
  if (check_flag(&retval, "ARKStepSetLinearSolver (main)", 1)) MPI_Abort(udata.comm, 1);
  if (!opts.iterative) {
    retval = ARKStepSetJacFn(arkode_mem, Jimpl);
    if (check_flag(&retval, "ARKStepSetJacFn (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set step postprocessing routine to update fluid energy (derived) field from other quantities
  retval = ARKStepSetPostprocessStepFn(arkode_mem, PostprocessStep);
  if (check_flag(&retval, "ARKStepSetPostprocessStepFn (main)", 1)) MPI_Abort(udata.comm, 1);

  // set diagnostics file
  if (udata.showstats && outproc) {
    retval = ARKStepSetDiagnostics(arkode_mem, DFID);
    if (check_flag(&retval, "ARKStepSetDiagnostics (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // set ARK Butcher tables
  retval = ARKStepSetTableNum(arkode_mem, ARK437L2SA_DIRK_7_3_4, ARK437L2SA_ERK_7_3_4);
  if (check_flag(&retval, "ARKStepSetTableNum (main)", 1)) MPI_Abort(udata.comm, 1);

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
    retval = ARKStepSetAdaptivityMethod(arkode_mem, opts.adapt_method,
                                        idefault, opts.pq, adapt_params);
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
  retval = ARKStepSVtolerances(arkode_mem, opts.rtol, atols);
  if (check_flag(&retval, "ARKStepSVtolerances (main)", 1)) MPI_Abort(udata.comm, 1);

  // set implicit predictor method
  retval = ARKStepSetPredictorMethod(arkode_mem, opts.predictor);
  if (check_flag(&retval, "ARKStepSetPredictorMethod (main)", 1)) MPI_Abort(udata.comm, 1);

  // set max nonlinear iterations
  retval = ARKStepSetMaxNonlinIters(arkode_mem, opts.maxniters);
  if (check_flag(&retval, "ARKStepSetMaxNonlinIters (main)", 1)) MPI_Abort(udata.comm, 1);

  // set nonlinear tolerance safety factor
  retval = ARKStepSetNonlinConvCoef(arkode_mem, opts.nlconvcoef);
  if (check_flag(&retval, "ARKStepSetNonlinConvCoef (main)", 1)) MPI_Abort(udata.comm, 1);


  // finish initialization
  realtype t = udata.t0;
  realtype tout = udata.t0+dTout;
  realtype hcur;
  if (opts.dense_order == -1)
    idense = 0;
  else   // otherwise tell integrator to use dense output
    idense = 1;


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
    retval = print_stats(t, w, 0, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  retval = unapply_Dengo_scaling(w, udata);
  if (check_flag(&retval, "unapply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
  //    Output normalized solution statistics (if requested)
  if (udata.showstats && PRINT_CGS == 0) {
    retval = print_stats(t, w, 0, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
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

    // adaptive evolution over [t0,t0+htrans]
    retval = ARKStepEvolve(arkode_mem, tout, w, &t, ARK_NORMAL);
    if (retval < 0) {    // unsuccessful solve: break
      if (outproc)  cerr << "Solver failure, stopping integration\n";
      cleanup(&arkode_mem, udata, BLS, LS, A, w, atols, wsubvecs, Nsubvecs);
      return(1);
    }

    // stop transient solver profiler
    retval = udata.profile[PR_TRANS].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

    if (outproc) {
      long int nst, nst_a, nfe, nfi, netf, nni, ncf;
      long int nls, nje, nli, nlcf;
      nst = nst_a = nfe = nfi = netf = nni = ncf = 0;
      nls = nje = nli = nlcf = 0;
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
      cout << "\nTransient portion of simulation complete:\n";
      cout << "   Solver steps = " << nst << " (attempted = " << nst_a << ")\n";
      cout << "   Total RHS evals:  Fe = " << nfe << ",  Fi = " << nfi << "\n";
      cout << "   Total number of error test failures = " << netf << "\n";
      if (opts.iterative && nli > 0) {
        cout << "   Total number of lin iters = " << nli << "\n";
        cout << "   Total number of lin conv fails = " << nlcf << "\n";
        cout << "   Total number of lin RHS evals = " << BDMPIMV_NFEDQ(LS) << "\n";
      } else if (nls > 0) {
        cout << "   Total number of lin solv setups = " << nls << "\n";
        cout << "   Total number of Jac eavls = " << nje << "\n";
      }
      if (nni > 0) {
        cout << "   Total number of nonlin iters = " << nni << "\n";
        cout << "   Total number of nonlin conv fails = " << ncf << "\n";
      }
      cout << "\nCurrent profiling results:\n";
    }
    retval = MPI_Barrier(udata.comm);
    if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
    udata.profile[PR_SETUP].print_cumulative_times("setup");
    udata.profile[PR_IO].print_cumulative_times("I/O");
    udata.profile[PR_MPI].print_cumulative_times("MPI");
    udata.profile[PR_PACKDATA].print_cumulative_times("pack");
    udata.profile[PR_FACEFLUX].print_cumulative_times("flux");
    udata.profile[PR_RHSEULER].print_cumulative_times("Euler RHS");
    udata.profile[PR_RHSSLOW].print_cumulative_times("explicit RHS");
    udata.profile[PR_RHSFAST].print_cumulative_times("implicit RHS");
    udata.profile[PR_JACFAST].print_cumulative_times("implicit Jac");
    udata.profile[PR_LSETUP].print_cumulative_times("lsetup");
    udata.profile[PR_LSOLVE].print_cumulative_times("lsolve");
    udata.profile[PR_LATIMES].print_cumulative_times("Atimes");
    udata.profile[PR_POSTFAST].print_cumulative_times("poststep");
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
    udata.profile[PR_POSTFAST].reset();
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
      retval = print_stats(t, w, 0, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
      if (PRINT_CGS == 1) {
        retval = unapply_Dengo_scaling(w, udata);
        if (check_flag(&retval, "unapply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
      }
    }
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);


    // disable adaptivity and set fixed step size
    retval = ARKStepSetFixedStep(arkode_mem, opts.hmax);
    if (check_flag(&retval, "ARKStepSetFixedStep (main)", 1)) MPI_Abort(udata.comm, 1);

  }


  //--- Main time-stepping loop: calls ARKStepEvolve to perform the integration, ---//
  //--- then prints results.  Stops when the final time has been reached.        ---//
  retval = udata.profile[PR_SIMUL].start();
  if (check_flag(&retval, "Profile::start (main)", 1)) MPI_Abort(udata.comm, 1);
  tout = udata.t0+dTout;
  for (int iout=restart; iout<restart+udata.nout; iout++) {

    // set stop time if applicable
    if (!idense) {
      retval = ARKStepSetStopTime(arkode_mem, tout);
      if (check_flag(&retval, "ARKStepSetStopTime (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    // evolve solution
    retval = ARKStepEvolve(arkode_mem, tout, w, &t, ARK_NORMAL);
    if (retval >= 0) {                         // successful solve: update output time
      tout = min(tout+dTout, udata.tf);
    } else {                                   // unsuccessful solve: break
      if (outproc)  cerr << "Solver failure, stopping integration\n";
      cleanup(&arkode_mem, udata, BLS, LS, A, w, atols, wsubvecs, Nsubvecs);
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
      retval = print_stats(t, w, 1, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    //    output results to disk -- get current step from ARKStep first
    retval = ARKStepGetLastStep(arkode_mem, &hcur);
    if (check_flag(&retval, "ARKStepGetLastStep (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = apply_Dengo_scaling(w, udata);
    if (check_flag(&retval, "apply_Dengo_scaling (main)", 1)) MPI_Abort(udata.comm, 1);
    if (NoOutput == 0) {
      retval = output_solution(t, w, hcur, iout+1, udata, opts);
      if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
    }
    //    output CGS statistics to stdout (if requested)
    if (udata.showstats && (PRINT_CGS == 1)) {
      retval = print_stats(t, w, 1, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
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
    retval = print_stats(t, w, 2, PRINT_SCIENTIFIC, PRINT_CGS, arkode_mem, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    retval = udata.profile[PR_IO].stop();
    if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  if (udata.showstats && outproc) {
    fclose(DFID);
  }

  // compute simulation time, total time
  retval = udata.profile[PR_SIMUL].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);
  retval = udata.profile[PR_TOTAL].stop();
  if (check_flag(&retval, "Profile::stop (main)", 1)) MPI_Abort(udata.comm, 1);

  // Print some final statistics
  if (outproc) {
    long int nst, nst_a, nfe, nfi, netf, nni, ncf;
    long int nls, nje, nli, nlcf;
    nst = nst_a = nfe = nfi = netf = nni = ncf = 0;
    nls = nje = nli = nlcf = 0;
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
    cout << "\nOverall Solver Statistics:\n";
    cout << "   Solver steps = " << nst << " (attempted = " << nst_a << ")\n";
    cout << "   Total RHS evals:  Fe = " << nfe << ",  Fi = " << nfi << "\n";
    cout << "   Total number of error test failures = " << netf << "\n";
    if (opts.iterative && nli > 0) {
      cout << "   Total number of lin iters = " << nli << "\n";
      cout << "   Total number of lin conv fails = " << nlcf << "\n";
      cout << "   Total number of lin RHS evals = " << BDMPIMV_NFEDQ(LS) << "\n";
    } else if (nls > 0) {
      cout << "   Total number of lin solv setups = " << nls << "\n";
      cout << "   Total number of Jac eavls = " << nje << "\n";
    }
    if (nni > 0) {
      cout << "   Total number of nonlin iters = " << nni << "\n";
      cout << "   Total number of nonlin conv fails = " << ncf << "\n";
    }
    cout << "\nFinal profiling results:\n";
  }
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  udata.profile[PR_SETUP].print_cumulative_times("setup");
  udata.profile[PR_IO].print_cumulative_times("I/O");
  udata.profile[PR_MPI].print_cumulative_times("MPI");
  udata.profile[PR_PACKDATA].print_cumulative_times("pack");
  udata.profile[PR_FACEFLUX].print_cumulative_times("flux");
  udata.profile[PR_RHSEULER].print_cumulative_times("Euler RHS");
  udata.profile[PR_RHSSLOW].print_cumulative_times("explicit RHS");
  udata.profile[PR_RHSFAST].print_cumulative_times("implicit RHS");
  udata.profile[PR_JACFAST].print_cumulative_times("implicit Jac");
  udata.profile[PR_LSETUP].print_cumulative_times("lsetup");
  udata.profile[PR_LSOLVE].print_cumulative_times("lsolve");
  udata.profile[PR_LATIMES].print_cumulative_times("Atimes");
  udata.profile[PR_POSTFAST].print_cumulative_times("poststep");
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
  cleanup(&arkode_mem, udata, BLS, LS, A, w, atols, wsubvecs, Nsubvecs);
#ifdef RAJA_CUDA
  // Destroy the cuSOLVER and cuSPARSE handles
  if (!opts.iterative) {
    cusparseDestroy(cusp_handle);
    cusolverSpDestroy(cusol_handle);
  }
#endif
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  MPI_Finalize();                  // Finalize MPI
  return 0;
}


//---- problem-defining functions (wrappers for other routines) ----

static int fimpl(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // start timer
  EulerData *udata = (EulerData*) user_data;
  int retval = udata->profile[PR_RHSFAST].start();
  if (check_flag(&retval, "Profile::start (fimpl)", 1)) return(-1);

  // initialize all outputs to zero (necessary!!)
  N_VConst(ZERO, wdot);

  // unpack chemistry subvectors
  N_Vector wchem = NULL;
  wchem = N_VGetSubvector_MPIManyVector(w, 5);
  if (check_flag((void *) wchem, "N_VGetSubvector_MPIManyVector (fimpl)", 0)) return(-1);
  N_Vector wchemdot = NULL;
  wchemdot = N_VGetSubvector_MPIManyVector(wdot, 5);
  if (check_flag((void *) wchemdot, "N_VGetSubvector_MPIManyVector (fimpl)", 0)) return(-1);

  // NOTE: if Dengo RHS ever does depend on fluid field inputs, those must
  // be converted to physical units prior to entry (via udata->DensityUnits, etc.)

  // call Dengo RHS routine
  retval = calculate_rhs_cvklu(t, wchem, wchemdot, udata->RxNetData);
  if (check_flag(&retval, "calculate_rhs_cvklu (fimpl)", 1)) return(retval);

  // NOTE: if fluid fields were rescaled to physical units above, they
  // must be converted back to code units here

  // scale wchemdot by TimeUnits to handle step size nondimensionalization
  N_VScale(udata->TimeUnits, wchemdot, wchemdot);

  // save chem RHS for use in ATimes (iterative linear solvers)
  if (udata->fchemcur) N_VScale(ONE, wchemdot, udata->fchemcur);

  // stop timer and return
  retval = udata->profile[PR_RHSFAST].stop();
  if (check_flag(&retval, "Profile::stop (fimpl)", 1)) return(-1);
  return(0);
}

static int Jimpl(realtype t, N_Vector w, N_Vector fw, SUNMatrix Jac,
                 void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // start timer
  EulerData *udata = (EulerData*) user_data;
  int retval = udata->profile[PR_JACFAST].start();
  if (check_flag(&retval, "Profile::start (Jimpl)", 1)) return(-1);

  // unpack chemistry subvectors
  N_Vector wchem = NULL;
  wchem = N_VGetSubvector_MPIManyVector(w, 5);
  if (check_flag((void *) wchem, "N_VGetSubvector_MPIManyVector (Jimpl)", 0)) return(-1);
  N_Vector fwchem = NULL;
  fwchem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) fwchem, "N_VGetSubvector_MPIManyVector (Jimpl)", 0)) return(-1);
  N_Vector tmp1chem = NULL;
  tmp1chem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) tmp1chem, "N_VGetSubvector_MPIManyVector (Jimpl)", 0)) return(-1);
  N_Vector tmp2chem = NULL;
  tmp2chem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) tmp2chem, "N_VGetSubvector_MPIManyVector (Jimpl)", 0)) return(-1);
  N_Vector tmp3chem = NULL;
  tmp3chem = N_VGetSubvector_MPIManyVector(fw, 5);
  if (check_flag((void *) tmp3chem, "N_VGetSubvector_MPIManyVector (Jimpl)", 0)) return(-1);

  // NOTE: if Dengo Jacobian ever does depend on fluid field inputs, those must
  // be converted to physical units prior to entry (via udata->DensityUnits, etc.)

  // call Dengo Jacobian routine
  retval = calculate_sparse_jacobian_cvklu(t, wchem, fwchem, Jac, udata->RxNetData,
                                           tmp1chem, tmp2chem, tmp3chem);

  // NOTE: if fluid fields were rescaled to physical units above, they
  // must be converted back to code units here

  // scale Jac values by TimeUnits to handle step size nondimensionalization
  realtype *Jdata = NULL;
  realtype TUnit = udata->TimeUnits;
#ifdef RAJA_CUDA
  Jdata = SUNMatrix_cuSparse_Data(Jac);
  if (check_flag((void *) Jdata, "SUNMatrix_cuSparse_Data (Jimpl)", 0)) return(-1);
  sunindextype nnz = SUNMatrix_cuSparse_NNZ(Jac);
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,nnz), [=] RAJA_DEVICE (long int i) {
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
  if (check_flag(&retval, "Profile::stop (Jimpl)", 1)) return(-1);
  return(0);
}

static int fexpl(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  long int i, j, k, l, cidx, fidx;

  // start timer
  EulerData *udata = (EulerData*) user_data;
  int retval = udata->profile[PR_RHSSLOW].start();
  if (check_flag(&retval, "Profile::start (fexpl)", 1)) return(-1);

  // initialize all outputs to zero (necessary??)
  N_VConst(ZERO, wdot);

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);
  realtype *etdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,4);
  if (check_flag((void *) etdot, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);
  realtype *chemdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,5);
  if (check_flag((void *) chemdot, "N_VGetSubvectorArrayPointer (fexpl)", 0)) return(-1);

  // update chem to include Dengo scaling
  retval = apply_Dengo_scaling(w, *udata);
  if (check_flag(&retval, "apply_Dengo_scaling (fexpl)", 1)) return(-1);

#ifdef RAJA_CUDA
  // ensure that chemistry data is synchronized to host
  N_VCopyFromDevice_Raja(N_VGetSubvector_MPIManyVector(w,5));
#endif

  // fill dimensionless total fluid energy field (internal energy + kinetic energy)
  realtype EUnitScale = ONE/udata->EnergyUnits;
  for (k=0; k<udata->nzl; k++)
    for (j=0; j<udata->nyl; j++)
      for (i=0; i<udata->nxl; i++) {
        cidx = BUFIDX(udata->nchem-1,i,j,k,udata->nchem,udata->nxl,udata->nyl,udata->nzl);
        realtype ge = chem[cidx];
        ge *= EUnitScale;   // convert from physical units to code units
        fidx = IDX(i,j,k,udata->nxl,udata->nyl,udata->nzl);
        et[fidx] = ge + 0.5/rho[fidx]*(mx[fidx]*mx[fidx] + my[fidx]*my[fidx] + mz[fidx]*mz[fidx]);
      }

#ifndef DISABLE_HYDRO
  // call fEuler as usual
  retval = fEuler(t, w, wdot, user_data);
  if (check_flag(&retval, "fEuler (fexpl)", 1)) return(retval);
#endif

  // overwrite chemistry energy "fexpl" with total energy "fexpl" (with
  // appropriate unit scaling) and zero out total energy fexpl
  //
  // QUESTION: is this really necessary, since fEuler also advects chemistry gas energy?
  // PARTIAL ANSWER: the external forces are currently only applied to the fluid fields,
  //   so these need to additionally force the chemistry gas energy
  //
  // Note: fEuler computes dy/dtau where tau = t / TimeUnits, but chemistry
  // RHS should compute dy/dt = dy/dtau * dtau/dt = dy/dtau * 1/TimeUnits
//  realtype TUnitScale = ONE/udata->TimeUnits;
  realtype TUnitScale = ONE;
  for (k=0; k<udata->nzl; k++)
    for (j=0; j<udata->nyl; j++)
      for (i=0; i<udata->nxl; i++) {
        cidx = BUFIDX(udata->nchem-1,i,j,k,udata->nchem,udata->nxl,udata->nyl,udata->nzl);
        fidx = IDX(i,j,k,udata->nxl,udata->nyl,udata->nzl);
        chemdot[cidx] = etdot[fidx]*TUnitScale;
        etdot[fidx] = ZERO;
      }

#ifdef RAJA_CUDA
  // ensure that chemistry rate-of-change data is synchronized back to device
  N_VCopyToDevice_Raja(N_VGetSubvector_MPIManyVector(wdot,5));
#endif

  // reset chem to remove Dengo scaling
  retval = unapply_Dengo_scaling(w, *udata);
  if (check_flag(&retval, "unapply_Dengo_scaling (fexpl)", 1)) return(-1);

  // stop timer and return
  retval = udata->profile[PR_RHSSLOW].stop();
  if (check_flag(&retval, "Profile::stop (fexpl)", 1)) return(-1);
  return(0);
}

static int PostprocessStep(realtype t, N_Vector w, void* user_data)
{
  // start timer
  EulerData *udata = (EulerData*) user_data;
  cvklu_data *network_data = (cvklu_data*) udata->RxNetData;
  int retval = udata->profile[PR_POSTFAST].start();
  if (check_flag(&retval, "Profile::start (PostprocessStep)", 1)) return(-1);

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (PostprocessStep)", 0)) return(-1);
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (PostprocessStep)", 0)) return(-1);
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (PostprocessStep)", 0)) return(-1);
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (PostprocessStep)", 0)) return(-1);
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (PostprocessStep)", 0)) return(-1);
  realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
  if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (PostprocessStep)", 0)) return(-1);

  // update chem to include Dengo scaling
  retval = apply_Dengo_scaling(w, *udata);
  if (check_flag(&retval, "apply_Dengo_scaling (PostprocessStep)", 1)) return(-1);

#ifdef RAJA_CUDA
  // ensure that chemistry data is synchronized to host
  N_VCopyFromDevice_Raja(N_VGetSubvector_MPIManyVector(w,5));
#endif

  // update fluid energy (derived) field from other quantities
  realtype EUnitScale = ONE/udata->EnergyUnits;
  for (int k=0; k<udata->nzl; k++)
    for (int j=0; j<udata->nyl; j++)
      for (int i=0; i<udata->nxl; i++) {
        const long int cidx = BUFIDX(udata->nchem-1,i,j,k,udata->nchem,udata->nxl,udata->nyl,udata->nzl);
        const long int fidx = IDX(i,j,k,udata->nxl,udata->nyl,udata->nzl);
        et[fidx] = chem[cidx] * EUnitScale
          + 0.5/rho[fidx]*(pow(mx[fidx],2) + pow(my[fidx],2) + pow(mz[fidx],2));
      }

  // reset chem to remove Dengo scaling
  retval = unapply_Dengo_scaling(w, *udata);
  if (check_flag(&retval, "unapply_Dengo_scaling (PostprocessStep)", 1)) return(-1);

  // stop timer and return
  retval = udata->profile[PR_POSTFAST].stop();
  if (check_flag(&retval, "Profile::stop (PostprocessStep)", 1)) return(-1);
  return(0);
}


//---- utility routines ----

void cleanup(void **arkode_mem, EulerData& udata, SUNLinearSolver BLS,
             SUNLinearSolver LS, SUNMatrix A, N_Vector w,
             N_Vector atols, N_Vector *wsubvecs, int Nsubvecs)
{
  ARKStepFree(arkode_mem);         // Free integrator memory
  SUNLinSolFree(BLS);              // Free matrix and linear solvers
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(w);                   // Free solution/tolerance vectors
  for (int i=0; i<Nsubvecs; i++)
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  N_VDestroy(atols);
  free_Dengo_structures(udata);
}


//---- custom SUNLinearSolver module ----

SUNLinearSolver SUNLinSol_BDMPIMV(SUNLinearSolver BLS, N_Vector x,
                                  sunindextype subvec, EulerData* udata,
                                  void* arkode_mem,
                                  ARKodeParameters& opts)
{
  // Check compatibility with supplied N_Vector
  if (N_VGetVectorID(x) != SUNDIALS_NVEC_MPIMANYVECTOR) return(NULL);
  if (subvec >= N_VGetNumSubvectors_MPIManyVector(x)) return(NULL);

  // Create an empty linear solver
  SUNLinearSolver S = SUNLinSolNewEmpty();
  if (S == NULL) return(NULL);

  // Attach operations (use defaults whenever possible)
  S->ops->gettype    = SUNLinSolGetType_BDMPIMV;
  S->ops->initialize = SUNLinSolInitialize_BDMPIMV;
  S->ops->setup      = SUNLinSolSetup_BDMPIMV;
  S->ops->solve      = SUNLinSolSolve_BDMPIMV;
  S->ops->lastflag   = SUNLinSolLastFlag_BDMPIMV;
  S->ops->setatimes  = SUNLinSolSetATimes_BDMPIMV;
  S->ops->free       = SUNLinSolFree_BDMPIMV;

  // Create, fill and attach content
  BDMPIMVContent content = NULL;
  content = (BDMPIMVContent) malloc(sizeof *content);
  if (content == NULL) { SUNLinSolFree(S); return(NULL); }

  content->blockLS    = BLS;
  content->subvec     = subvec;
  content->udata      = udata;
  content->arkode_mem = arkode_mem;
  content->nfeDQ      = 0;
  if (opts.iterative) {
    content->work   = N_VClone(x);
    udata->fchemcur = N_VClone(N_VGetSubvector_MPIManyVector(x, subvec));
  } else {
    content->work   = NULL;
    udata->fchemcur = NULL;
  }
  S->content = content;

  return(S);
}

SUNLinearSolver_Type SUNLinSolGetType_BDMPIMV(SUNLinearSolver S)
{
  if (BDMPIMV_WORK(S))
    return(SUNLINEARSOLVER_ITERATIVE);
  else
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
  int retval = BDMPIMV_UDATA(S)->profile[PR_LSETUP].start();
  if (check_flag(&retval, "Profile::start (SUNLinSolSetup_BDMPIMV)", 1))  return(retval);
  BDMPIMV_LASTFLAG(S) = SUNLinSolSetup(BDMPIMV_BLS(S), A);
  retval = BDMPIMV_UDATA(S)->profile[PR_LSETUP].stop();
  if (check_flag(&retval, "Profile::stop (SUNLinSolSetup_BDMPIMV)", 1))  return(retval);
  return(BDMPIMV_LASTFLAG(S));
}

int SUNLinSolSolve_BDMPIMV(SUNLinearSolver S, SUNMatrix A,
                           N_Vector x, N_Vector b, realtype tol)
{
  // start profiling timer
  int retval = BDMPIMV_UDATA(S)->profile[PR_LSOLVE].start();
  if (check_flag(&retval, "Profile::start (SUNLinSolSolve_BDMPIMV)", 1))  return(retval);

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
  retval = BDMPIMV_UDATA(S)->profile[PR_LSOLVE].stop();
  if (check_flag(&retval, "Profile::stop (SUNLinSolSolve_BDMPIMV)", 1))  return(retval);
  return(BDMPIMV_LASTFLAG(S));
}

int SUNLinSolSetATimes_BDMPIMV(SUNLinearSolver S, void* A_data, ATimesFn ATimes)
{
  // Ignore the input ARKODE ATimes function and attach a custom ATimes function
  BDMPIMV_LASTFLAG(S) = SUNLinSolSetATimes(BDMPIMV_BLS(S), BDMPIMV_CONTENT(S),
                                           ATimes_BDMPIMV);
  return(BDMPIMV_LASTFLAG(S));
}

int ATimes_BDMPIMV(void* A_data, N_Vector v, N_Vector z)
{
  // Access the linear solver content
  BDMPIMVContent content = (BDMPIMVContent) A_data;

  // Shortcuts to content
  EulerData* udata      = content->udata;
  void*      arkode_mem = content->arkode_mem;

  // Get the current time, gamma, and error weights
  realtype tcur;
  int retval = ARKStepGetCurrentTime(arkode_mem, &tcur);
  if (check_flag(&retval, "ARKStepGetCurrentTime (Atimes)", 1)) return(-1);

  N_Vector ycur;
  retval = ARKStepGetCurrentState(arkode_mem, &ycur);
  if (check_flag(&retval, "ARKStepGetCurrentState (Atimes)", 1)) return(-1);

  realtype gamma;
  retval = ARKStepGetCurrentGamma(arkode_mem, &gamma);
  if (check_flag(&retval, "ARKStepGetCurrentGamma (Atimes)", 1)) return(-1);

  N_Vector work = content->work;
  retval = ARKStepGetErrWeights(arkode_mem, work);
  if (check_flag(&retval, "ARKStepGetErrWeights (Atimes)", 1)) return(-1);

  // Get ycur and weight vector for chem species
  N_Vector y = N_VGetSubvector_MPIManyVector(ycur, content->subvec);
  N_Vector w = N_VGetSubvector_MPIManyVector(work, content->subvec);

  // Start timer
  retval = udata->profile[PR_LATIMES].start();
  if (check_flag(&retval, "Profile::start (Atimes)", 1)) return(-1);

  // Set perturbation to 1/||v||
  realtype sig = ONE / N_VWrmsNorm(v, w);

  // Set work = y + sig * v
  N_VLinearSum(sig, v, ONE, y, w);

  // Set z = fchem(t, y + sig * v)
  retval = calculate_rhs_cvklu(tcur, w, z, udata->RxNetData);
  content->nfeDQ++;
  if (check_flag(&retval, "calculate_rhs_cvklu (Atimes)", 1)) return(retval);

  // scale wchemdot by TimeUnits to handle step size nondimensionalization
  N_VScale(udata->TimeUnits, z, z);

  // Compute Jv approximation: z = (z - fchemcur) / sig
  realtype siginv = ONE / sig;
  N_VLinearSum(siginv, z, -siginv, udata->fchemcur, z);

  // Compute Av approximation: z = (I - gamma J) v
  N_VLinearSum(ONE, v, -gamma, z, z);

  // Stop timer and return
  retval = udata->profile[PR_LATIMES].stop();
  if (check_flag(&retval, "Profile::stop (Atimes)", 1)) return(-1);

  return(0);
}

sunindextype SUNLinSolLastFlag_BDMPIMV(SUNLinearSolver S)
{
  return(BDMPIMV_LASTFLAG(S));
}


int SUNLinSolFree_BDMPIMV(SUNLinearSolver S)
{
  BDMPIMVContent content = BDMPIMV_CONTENT(S);
  if (content == NULL) return(0);
  if (content->work) N_VDestroy(content->work);
  free(S->ops);
  free(S->content);
  free(S);
  return(0);
}

//---- end of file ----
