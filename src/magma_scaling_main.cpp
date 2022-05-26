/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2022, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Magma Dense solver scaling test:

 This program merely allocates a RAJA vector and the MagmaDense
 SUNMatrix and SUNLinearSolver objects separately on each MPI
 rank and exits.  Each phase of this is thoroughly timed, so that
 we can scale it up to examine any performance bottlenecks.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <arkode/arkode_arkstep.h>
#ifdef USERAJA
#include <nvector/nvector_raja.h>
#else
#error "magma_scaling.exe requires RAJA to be enabled; aborting build."
#endif
#ifdef USEMAGMA
#include <sunmatrix/sunmatrix_magmadense.h>
#include <sunlinsol/sunlinsol_magmadense.h>
#else
#error "magma_scaling.exe requires MAGMA to be enabled; aborting build."
#endif


// Utility function prototypes
int check_flag(const void *flagvalue, const string funcname, const int opt);

// Main Program
int main(int argc, char* argv[]) {

  // general problem variables
  long int N;
  int retval;                    // reusable error-checking flag
  int myid;                      // MPI rank ID
  int restart;                   // restart file number to use (disabled if negative)
  N_Vector w = NULL;             // empty vector for storing overall solution
  SUNMatrix A = NULL;            // empty matrix and linear solver structures
  SUNLinearSolver LS = NULL;
  ARKODEParameters opts;
  SUNProfiler profobj = NULL;    // empty profiler object

  //--- General Initialization ---//

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return(1);
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // declare user data structure once MPI has been initialized
  EulerData udata;

  // initial setup
  retval = MPI_Barrier(MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);
  SUNDIALS_MARK_BEGIN(profobj, "InitSetup");
  if (myid == 0)  cout << "Initializing problem\n";
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);
  bool outproc = (udata.myid == 0);
  retval = SUNContext_GetProfiler(udata.ctx, &profobj);
  if(check_flag(&retval, "SUNContext_GetProfiler", 1)) MPI_Abort(udata.comm, 1);

  SUNDIALS_MARK_FUNCTION_BEGIN(profobj);

  // set up overall grid parameters
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  SUNDIALS_MARK_BEGIN(profobj, "SetupDecomp");
  udata.nxl = udata.nyl = udata.nzl = 25;
  N = (udata.nxl)*(udata.nyl)*(udata.nzl);
  udata.nchem = 10;
  if (outproc) {
    cout << "\nMagma scaling driver:\n";
    cout << "   nprocs: " << udata.nprocs;
    cout << "   num chemical species: " << udata.nchem << "\n";
    cout << "   spatial grid: " << udata.nxl*udata.nprocs << " x " << udata.nyl << " x " << udata.nzl << "\n";
    cout << "   overall problem size: " << N*udata.nprocs*udata.nchem << "\n";
  }
  SUNDIALS_MARK_END(profobj, "SetupDecomp");

  // Initialize N_Vector data structures with configured vector operations
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  SUNDIALS_MARK_BEGIN(profobj, "NVectorInit");
  if (myid == 0)  cout << "Initializing N_Vector\n";
  w = N_VNewManaged_Raja(N*udata.nchem, udata.ctx);
  if (check_flag((void *) w, "N_VNewManaged_Raja (main)", 0)) MPI_Abort(udata.comm, 1);
  N_VConst(0.0, w);
  SUNDIALS_MARK_END(profobj, "NVectorInit");

  // Create SUNMatrix for use in linear solves
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  SUNDIALS_MARK_BEGIN(profobj, "SUNMatrixInit");
  if (myid == 0)  cout << "Initializing SUNMatrix\n";
  A = SUNMatrix_MagmaDenseBlock(N, udata.nchem, udata.nchem, SUNMEMTYPE_DEVICE,
                                udata.memhelper, NULL, udata.ctx);
  if(check_flag((void *) A, "SUNMatrix_MagmaDenseBlock", 0)) return(1);
  SUNDIALS_MARK_END(profobj, "SUNMatrixInit");

  // Create the SUNLinearSolver object
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  SUNDIALS_MARK_BEGIN(profobj, "SUNLinSolInit");
  if (myid == 0)  cout << "Initializing SUNLinSol\n";
  LS = SUNLinSol_MagmaDense(w, A, udata.ctx);
  if(check_flag((void *) LS, "SUNLinSol_MagmaDense", 0)) return(1);
  SUNDIALS_MARK_END(profobj, "SUNLinSolInit");

  // Output all profiling results
  SUNDIALS_MARK_FUNCTION_END(profobj);
  if (myid == 0)  cout << "Overall profiling results:\n";
  // retval = SUNProfiler_Print(profobj, stdout);
  // if (check_flag(&retval, "SUNProfiler_Print (main)", 1)) MPI_Abort(udata.comm, 1);

  // Clean up, finalize MPI, and return with successful completion
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(w);
  return 0;
}


// Required auxiliary routine (unused)
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  // initialize external forces to zero
  N_VConst(ZERO, G);
  return 0;
}


//---- end of file ----
