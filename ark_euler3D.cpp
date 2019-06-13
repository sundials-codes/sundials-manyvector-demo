/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Example problem:

 The following test simulates a 3D nonlinear inviscid
 compressible Euler equation,
    w_t = - Div(F(w)) + G,
 for t in [t0, tf], X = (x,y,z) in [xl,xr] x [yl,yr] x [zl,zr],
 with initial condition
    w(t0,X) = w0(X).
 Here, the state w = [rho, rho*vx, rho*vy, rho*vz, E]^T
                   = [rho, mx, my, mz, E]^T,
 corresponding to the density, x,y,z-momentum, and the total
 energy per unit volume.  The fluxes are given by
    Fx(w) = [rho*vx, rho*vx^2+p, rho*vx*vy, rho*vx*vz, vx*(E+p)]^T
    Fy(w) = [rho*vy, rho*vx*vy, rho*vy^2+p, rho*vy*vz, vy*(E+p)]^T
    Fz(w) = [rho*vz, rho*vx*vz, rho*vy*vz, rho*vz^2+p, vz*(E+p)]^T
 the external force is given by
    G(X) = [0, gx(X), gy(X), gz(X), 0]^T
 and the ideal gas equation of state gives
    p = R/cv*(E - rho/2*(vx^2+vy^2+vz^2), and
    E = p*cv/R + rho/2*(vx^2+vy^2+vz^2),
 or equivalently,
    p = (gamma-1)*(E - rho/2*(vx^2+vy^2+vz^2), and
    E = p/(gamma-1) + rho/2*(vx^2+vy^2+vz^2),
 We have the parameters
    R is the specific ideal gas constant (287.14 J/kg/K).
    cv is the specific heat capacity at constant volume (717.5 J/kg/K),
    gamma is the ratio of specific heats, cp/cv = 1 + R/cv (1.4),
 corresponding to air (predominantly an ideal diatomic gas).
 The speed of sound in the gas is then given by
    c = sqrt(gamma*p/rho).

 The fluid variables above are non-dimensionalized; in standard SI
 units these would be:
    [rho] = kg / m^3
    [vx] = [vy] = [vz] = m/s  =>  [rho*vx] = kg / m^2 / s
    [E] = kg / m / s^2

 Note: the above follows the description in section 7.3.1-7.3.3 of
   https://www.theoretical-physics.net/dev/fluid-dynamics/euler.html

 This program solves the problem using an ERK method.  100 outputs
 are printed at equal intervals, and run statistics are printed
 at the end.
---------------------------------------------------------------*/

// Header files
#include <stdio.h>
#include <iostream>
#include <utility>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <arkode/arkode_arkstep.h>
#include <nvector/nvector_mpimanyvector.h>
#include <nvector/nvector_parallel.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <mpi.h>

using namespace std;

// accessor macro between (x,y,z) location and 1D NVector array
#define IDX(x,y,z,nx,ny) ( (x) + (nx)*((y) + (ny)*(z)) )

// accessor macro between (v,x,y,z) location and 1D NVector array
#define BUFIDX(v,x,y,z,nx,ny,nz) ( (v) + 5*((x) + (nx)*((y) + (ny)*(z))) )

// MPI equivalent of realtype type
#if defined(SUNDIALS_SINGLE_PRECISION)
#define MPI_SUNREALTYPE MPI_FLOAT
#elif defined(SUNDIALS_DOUBLE_PRECISION)
#define MPI_SUNREALTYPE MPI_DOUBLE
#elif defined(SUNDIALS_EXTENDED_PRECISION)
#define MPI_SUNREALTYPE MPI_LONG_DOUBLE
#endif

// reused constants
#define ZERO    RCONST(0.0)
#define HALF    RCONST(0.5)
#define FOURTH  RCONST(0.25)
#define ONE     RCONST(1.0)
#define TWO     RCONST(2.0)
#define THREE   RCONST(3.0)
#define FOUR    RCONST(4.0)
#define FIVE    RCONST(5.0)
#define SEVEN   RCONST(7.0)
#define ELEVEN  RCONST(11.0)

// user data class
class UserData {

public:
  ///// [sub]domain related data /////
  long int nx;          // global number of x grid points
  long int ny;          // global number of y grid points
  long int nz;          // global number of z grid points
  long int is;          // global x indices of this subdomain
  long int ie;
  long int js;          // global y indices of this subdomain
  long int je;
  long int ks;          // global y indices of this subdomain
  long int ke;
  long int nxl;         // local number of x grid points
  long int nyl;         // local number of y grid points
  long int nzl;         // local number of z grid points
  realtype xl;          // spatial domain extents
  realtype xr;
  realtype yl;
  realtype yr;
  realtype zl;
  realtype zr;
  realtype dx;          // x-directional mesh spacing
  realtype dy;          // y-directional mesh spacing
  realtype dz;          // z-directional mesh spacing

  ///// problem-defining data /////
  int      xlbc;        // boundary condition types:
  int      xrbc;        //      0 = periodic
  int      ylbc;        //      1 = homogeneous Neumann
  int      yrbc;        //      2 = homogeneous Dirichlet
  int      zlbc;
  int      zrbc;
  realtype gamma;       // ratio of specific heat capacities, cp/cv

  ///// MPI-specific data /////
  MPI_Comm comm;        // communicator object
  int myid;             // MPI process ID
  int nprocs;           // total number of MPI processes
  int ipW;              // MPI ranks for neighbor procs
  int ipE;
  int ipS;
  int ipN;
  int ipB;
  int ipF;
  realtype *Erecv;      // receive buffers for neighbor exchange
  realtype *Wrecv;
  realtype *Nrecv;
  realtype *Srecv;
  realtype *Frecv;
  realtype *Brecv;
  realtype *Esend;      // send buffers for neighbor exchange
  realtype *Wsend;
  realtype *Nsend;
  realtype *Ssend;
  realtype *Fsend;
  realtype *Bsend;
  MPI_Request req[12];  // MPI requests for neighbor exchange

  ///// class operations /////
  // constructor
  UserData(long int nx_, long int ny_, long int nz_, realtype xl_,
           realtype xr_, realtype yl_, realtype yr_,
           realtype zl_, realtype zr_, int xlbc_, int xrbc_,
           int ylbc_, int yrbc_, int zlbc_, int zrbc_, realtype gamma_) :
      nx(nx_), ny(ny_), nz(nz_), xl(xl_), xr(xr_), yl(yl_), yr(yr_),
      zl(zl_), zr(zr_), xlbc(xlbc_), xrbc(xrbc_), ylbc(ylbc_), yrbc(yrbc_),
      zlbc(zlbc_), zrbc(zrbc_), is(0), ie(0), js(0), je(0), ks(0), ke(0),
      nxl(0), nyl(0), nzl(0), comm(MPI_COMM_WORLD), myid(0), nprocs(0),
      Erecv(NULL), Wrecv(NULL), Nrecv(NULL), Srecv(NULL), Frecv(NULL), Brecv(NULL),
      Esend(NULL), Wsend(NULL), Nsend(NULL), Ssend(NULL), Fsend(NULL), Bsend(NULL),
      ipW(-1), ipE(-1), ipS(-1), ipN(-1), ipB(-1), ipF(-1), gamma(gamma_)
  {
    dx = (xr-xl)/(nx-1);
    dy = (yr-yl)/(ny-1);
    dz = (zr-zl)/(nz-1);
  };
  // destructor
  ~UserData() {
    int i;
    if (Wrecv != NULL)  delete[] Wrecv;
    if (Wsend != NULL)  delete[] Wsend;
    if (Erecv != NULL)  delete[] Erecv;
    if (Esend != NULL)  delete[] Esend;
    if (Srecv != NULL)  delete[] Srecv;
    if (Ssend != NULL)  delete[] Ssend;
    if (Nrecv != NULL)  delete[] Nrecv;
    if (Nsend != NULL)  delete[] Nsend;
    if (Brecv != NULL)  delete[] Brecv;
    if (Bsend != NULL)  delete[] Bsend;
    if (Frecv != NULL)  delete[] Frecv;
    if (Fsend != NULL)  delete[] Fsend;
  };
  int SetupDecomp();              // set up parallel decomposition
  int ExchangeStart(N_Vector w);  // begin neighbor exchange
  int ExchangeEnd();              // finish neighbor exchange

};   // end UserData;

// User-supplied Functions Called by the Solver
static int f(realtype t, N_Vector w, N_Vector wdot, void* user_data);

// Private functions
//    Load inputs from file
int load_inputs(int myid, double& xl, double& xr, double& yl,
                double& yr, double& zl, double& zr, double& t0,
                double& tf, double& gamma, long int& nx, long int& ny,
                long int& nz, long int& xlbc, long int& xrbc,
                long int& ylbc, long int& yrbc, long int& zlbc,
                long int& zrbc, long int& showstats);
//    Equation of state
inline realtype eos(realtype& rho, realtype& mx, realtype& my,
                    realtype& mz, realtype& E, UserData& udata);
//    Check for legal state
inline int legal_state(realtype& rho, realtype& mx, realtype& my,
                       realtype& mz, realtype& E, UserData& udata);
//    Initial conditions
int initial_conditions(realtype t, N_Vector w, UserData& udata);
//    Forcing terms
int external_forces(realtype t, N_Vector wdot, UserData& udata);
//    Signed fluxes over patch
void signed_fluxes(realtype (&w1d)[7][5], realtype (&fms)[7][5],
                   realtype (&fps)[7][5], UserData& udata);
//    Print solution statistics
int print_stats(realtype t, N_Vector w, int firstlast, UserData& udata);
//    Output current solution
int output_solution(N_Vector w, int newappend, UserData& udata);
//    1D packing routines
inline void pack1D_x(realtype (&w1d)[7][5], realtype* rho, realtype* mx,
                     realtype* my, realtype* mz, realtype* E,
                     realtype* Wrecv, realtype* Erecv,
                     long int& i, long int& j, long int& k,
                     long int& nxl, long int& nyl, long int& nzl);
inline void pack1D_y(realtype (&w1d)[7][5], realtype* rho, realtype* mx,
                     realtype* my, realtype* mz, realtype* E,
                     realtype* Srecv, realtype* Nrecv,
                     long int& i, long int& j,  long int& k,
                     long int& nxl, long int& nyl, long int& nzl);
inline void pack1D_z(realtype (&w1d)[7][5], realtype* rho, realtype* mx,
                     realtype* my, realtype* mz, realtype* E,
                     realtype* Brecv, realtype* Frecv,
                     long int& i, long int& j, long int& k,
                     long int& nxl, long int& nyl, long int& nzl);
//    WENO Div(flux(u)) function
void div_flux(realtype (&w1d)[7][5], int idir, realtype& dx, realtype* dw, UserData& udata);
//    checks function return values
int check_flag(void *flagvalue, const string funcname, int opt);
//    Parameter input helper function
void* arkstep_init_from_file(char *fname, ARKRhsFn f, ARKRhsFn fe, ARKRhsFn fi,
                             realtype t0, N_Vector w0, int *ImEx, int *dorder,
                             int *fxpt, realtype *RTol, realtype *ATol);


// Main Program
int main(int argc, char* argv[]) {

  // general problem parameters
  int Nt = 100;                  // total number of output times
  long int N, Ntot, i;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int dense_order;               // dense output order of accuracy
  int idense;                    // flag denoting integration type (dense output vs tstop)
  int imex;                      // flag denoting class of method (0=implicit, 1=explicit, 2=IMEX)
  int fixedpt;                   // flag denoting use of fixed-point nonlinear solver
  int myid;                      // MPI process ID
  N_Vector w = NULL;             // empty vectors for storing overall solution
  N_Vector wsub[5] = {NULL, NULL, NULL, NULL, NULL};
  void *arkode_mem = NULL;       // empty ARKStep memory structure

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // start run timer
  double tstart = MPI_Wtime();

  /* read problem parameters from input file */
  double xl, xr, yl, yr, zl, zr, t0, tf, gamma;
  long int nx, ny, nz, xlbc, xrbc, ylbc, yrbc, zlbc, zrbc, showstats;
  retval = load_inputs(myid, xl, xr, yl, yr, zl, zr, t0, tf, gamma, nx, ny, nz,
                       xlbc, xrbc, ylbc, yrbc, zlbc, zrbc, showstats);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // allocate and fill udata structure (overwriting betax for this test)
  UserData udata(nx,ny,nz,xl,xr,yl,yr,zl,zr,xlbc,xrbc,ylbc,yrbc,zlbc,zrbc,gamma);
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);

  // Initial problem output
  bool outproc = (udata.myid == 0);
  if (outproc) {
    cout << "\n3D compressible inviscid Euler equations test problem:\n";
    cout << "   nprocs = " << udata.nprocs << " ( " << udata.nx << " x "
         << udata.ny << " x " << udata.nz << " )\n";
    cout << "   spatial domain = [ " << udata.xl << " , " << udata.xr << " ] x [ "
         << udata.yl << " , " << udata.yr << " ] x [ "
         << udata.zl << " , " << udata.zr << " ]\n";
    cout << "   time domain = ( " << t0 << " , " << tf << " ]\n";
    cout << "   bdry cond (0=per, 1=Neu, 2=Dir) = [ "
         << udata.xlbc << " , " << udata.xrbc << " ] x [ "
         << udata.ylbc << " , " << udata.yrbc << " ] x [ "
         << udata.zlbc << " , " << udata.zrbc << " ]\n";
    cout << "   gamma = " << udata.gamma << "\n";
  }
  if (showstats)
    cout << "   proc " << udata.myid << ": (nxl,nyl,nzl) = ( "
         << udata.nxl << " , " << udata.nyl << " , " << udata.nzl << " )\n";

  // open solver diagnostics output file for writing
  FILE *DFID = NULL;
  if (outproc)
    DFID=fopen("diags_ark_euler3D.txt","w");

  // Initialize N_Vector data structures
  N = (udata.nxl)*(udata.nyl)*(udata.nzl)*5;
  Ntot = nx*ny*nz*5;
  for (i=0; i<5; i++) {
    wsub[i] = N_VNew_Parallel(udata.comm, N, Ntot);
    if (check_flag((void *) wsub[i], "N_VNew_Parallel (main)", 0)) MPI_Abort(udata.comm, 1);
  }
  w = N_VNew_MPIManyVector(5, wsub);  // solution vector
  if (check_flag((void *) w, "N_VNew_MPIManyVector (main)", 0)) MPI_Abort(udata.comm, 1);

  // set initial conditions
  retval = initial_conditions(t0, w, udata);         // Set initial conditions
  if (check_flag(&retval, "initial_conditions (main)", 1)) MPI_Abort(udata.comm, 1);

  // Call arkstep_init_from_file helper routine to read and set solver parameters;
  // quit if input file disagrees with desired solver options
  realtype rtol, atol;
  arkode_mem = arkstep_init_from_file("solve_params.txt", f, NULL, NULL, t0, w,
                                      &imex, &dense_order, &fixedpt, &rtol, &atol);
  if (check_flag((void *) arkode_mem, "arkstep_init_from_file (main)", 1)) MPI_Abort(udata.comm, 1);
  if (rtol <= 0.0)  rtol = 1.e-6;
  if (atol <= 0.0)  atol = 1.e-10;

  // Return error message if an implicit solver is requested
  if (imex != 1) {
    if (outproc)
      cerr << "Error: solve_params.txt requested an implicit or imex solver\n"
           << "  (only explicit methods are currently supported)\n";
    MPI_Abort(udata.comm, 1);
  }

  // If (dense_order == -1), tell integrator to use tstop
  if (dense_order == -1)
    idense = 0;
  else          // otherwise tell integrator to use dense output
    idense = 1;

  // Set routines
  retval = ARKStepSetUserData(arkode_mem, (void *) (&udata));   // Pass udata to user functions
  if (check_flag(&retval, "ARKStepSetUserData (main)", 1)) MPI_Abort(udata.comm, 1);
  if (outproc) {
    retval = ARKStepSetDiagnostics(arkode_mem, DFID);           // Set diagnostics file
    if (check_flag(&retval, "ARKStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  retval = ARKStepSStolerances(arkode_mem, rtol, atol);         // Specify tolerances
  if (check_flag(&retval, "ARKStepSStolerances (main)", 1)) MPI_Abort(udata.comm, 1);

  // Each processor outputs subdomain information
  char outname[100];
  sprintf(outname, "euler3d_subdomain.%03i.txt", udata.myid);
  FILE *UFID = fopen(outname,"w");
  fprintf(UFID, "%li  %li  %li  %li  %li  %li  %li  %li  %li\n",
	  udata.nx, udata.ny, udata.nz, udata.is, udata.ie,
          udata.js, udata.je, udata.ks, udata.ke);
  fclose(UFID);

  // Output initial conditions to disk
  retval = output_solution(w, 1, udata);
  if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);

  // compute setup time
  double tsetup = MPI_Wtime() - tstart;
  tstart = MPI_Wtime();

  /* Main time-stepping loop: calls ARKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  realtype t = t0;
  realtype dTout = (tf-t0)/Nt;
  realtype tout = t0+dTout;
  if (showstats) {
    retval = print_stats(t, w, 0, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  int iout;
  for (iout=0; iout<Nt; iout++) {

    if (!idense)
      retval = ARKStepSetStopTime(arkode_mem, tout);
    retval = ARKStepEvolve(arkode_mem, tout, w, &t, ARK_NORMAL);  // call integrator
    if (retval >= 0) {                                            // successful solve: update output time
      tout = min(tout+dTout, tf);
    } else {                                                      // unsuccessful solve: break
      if (outproc)
	cerr << "Solver failure, stopping integration\n";
      return 1;
    }

    // output statistics to stdout
    if (showstats) {
      retval = print_stats(t, w, 1, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    // output results to disk
    retval = output_solution(w, 0, udata);
    if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  if (showstats) {
    retval = print_stats(t, w, 2, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
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
    cout << "   Total simulation time = " << tsimul << "\n";
  }

  // Clean up and return with successful completion
  N_VDestroy(w);               // Free solution vectors
  for (i=0; i<5; i++)
    N_VDestroy(wsub[i]);
  ARKStepFree(&arkode_mem);    // Free integrator memory
  MPI_Finalize();              // Finalize MPI
  return 0;
}

/*--------------------------------
 * Functions called by the solver
 *--------------------------------*/

// f routine to compute the ODE RHS function f(t,y).
static int f(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // access problem data
  UserData *udata = (UserData *) user_data;

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *E = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) E, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *rhodot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,0);
  if (check_flag((void *) rhodot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mxdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,1);
  if (check_flag((void *) mxdot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mydot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,2);
  if (check_flag((void *) mydot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mzdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,3);
  if (check_flag((void *) mzdot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *Edot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,4);
  if (check_flag((void *) Edot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;

  // Exchange boundary data with neighbors
  int retval = udata->ExchangeStart(w);
  if (check_flag(&retval, "ExchangeStart (f)", 1)) return -1;

  // Initialize wdot with external forcing terms
  retval = external_forces(t, wdot, *udata);
  if (check_flag(&retval, "external_forces (f)", 1)) return -1;

  // Set shortcut variables
  long int nxl = udata->nxl;
  long int nyl = udata->nyl;
  long int nzl = udata->nzl;

  // iterate over subdomain interior, computing approximation to RHS
  realtype w1d[7][5], dw[5];
  long int v, i, j, k;
  for (k=3; k<nzl-3; k++)
    for (j=3; j<nyl-3; j++)
      for (i=3; i<nxl-3; i++) {

        // return with failure on non-positive density, energy or pressure
        retval = legal_state(rho[IDX(i,j,k,nxl,nyl)], mx[IDX(i,j,k,nxl,nyl)],
                             my[IDX(i,j,k,nxl,nyl)], mz[IDX(i,j,k,nxl,nyl)],
                             E[IDX(i,j,k,nxl,nyl)], *udata);
        if (check_flag(&retval, "legal_state (f)", 1)) return -1;

        // x-directional advection
        //    pack 1D array of variable shortcuts
        pack1D_x(w1d, rho, mx, my, mz, E, udata->Wrecv, udata->Erecv, i, j, k, nxl, nyl, nzl);
        //    compute update
        div_flux(w1d, 0, udata->dx, dw, *udata);
        //    apply update
        rhodot[IDX(i,j,k,nxl,nyl)] -= dw[0];
        mxdot[ IDX(i,j,k,nxl,nyl)] -= dw[1];
        mydot[ IDX(i,j,k,nxl,nyl)] -= dw[2];
        mzdot[ IDX(i,j,k,nxl,nyl)] -= dw[3];
        Edot[  IDX(i,j,k,nxl,nyl)] -= dw[4];

        // y-directional advection
        //    pack 1D array of variable shortcuts
        pack1D_y(w1d, rho, mx, my, mz, E, udata->Srecv, udata->Nrecv, i, j, k, nxl, nyl, nzl);
        //    compute update
        div_flux(w1d, 1, udata->dy, dw, *udata);
        //    apply update
        rhodot[IDX(i,j,k,nxl,nyl)] -= dw[0];
        mxdot[ IDX(i,j,k,nxl,nyl)] -= dw[1];
        mydot[ IDX(i,j,k,nxl,nyl)] -= dw[2];
        mzdot[ IDX(i,j,k,nxl,nyl)] -= dw[3];
        Edot[  IDX(i,j,k,nxl,nyl)] -= dw[4];

        // z-directional advection
        //    pack 1D array of variable shortcuts
        pack1D_z(w1d, rho, mx, my, mz, E, udata->Brecv, udata->Frecv, i, j, k, nxl, nyl, nzl);
        //    compute update
        div_flux(w1d, 2, udata->dz, dw, *udata);
        //    apply update
        rhodot[IDX(i,j,k,nxl,nyl)] -= dw[0];
        mxdot[ IDX(i,j,k,nxl,nyl)] -= dw[1];
        mydot[ IDX(i,j,k,nxl,nyl)] -= dw[2];
        mzdot[ IDX(i,j,k,nxl,nyl)] -= dw[3];
        Edot[  IDX(i,j,k,nxl,nyl)] -= dw[4];

      }

  // wait for boundary data to arrive from neighbors
  retval = udata->ExchangeEnd();
  if (check_flag(&retval, "ExchangeEnd (f)", 1)) return -1;

  // iterate over entire domain, skipping interior
  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {

        // skip strict interior (already computed)
        if ( (k>2) && (k<nzl-3) &&
             (j>2) && (j<nyl-3) &&
             (i>2) && (i<nxl-3) ) continue;

        // return with failure on non-positive density, energy or pressure
        retval = legal_state(rho[IDX(i,j,k,nxl,nyl)], mx[IDX(i,j,k,nxl,nyl)],
                             my[IDX(i,j,k,nxl,nyl)], mz[IDX(i,j,k,nxl,nyl)],
                             E[IDX(i,j,k,nxl,nyl)], *udata);
        if (check_flag(&retval, "legal_state (f)", 1)) return -1;

        // x-directional advection
        //    pack 1D array of variable shortcuts
        pack1D_x(w1d, rho, mx, my, mz, E, udata->Wrecv, udata->Erecv, i, j, k, nxl, nyl, nzl);
        //    compute update
        div_flux(w1d, 0, udata->dx, dw, *udata);
        //    apply update
        rhodot[IDX(i,j,k,nxl,nyl)] -= dw[0];
        mxdot[ IDX(i,j,k,nxl,nyl)] -= dw[1];
        mydot[ IDX(i,j,k,nxl,nyl)] -= dw[2];
        mzdot[ IDX(i,j,k,nxl,nyl)] -= dw[3];
        Edot[  IDX(i,j,k,nxl,nyl)] -= dw[4];

        // y-directional advection
        //    pack 1D array of variable shortcuts
        pack1D_y(w1d, rho, mx, my, mz, E, udata->Srecv, udata->Nrecv, i, j, k, nxl, nyl, nzl);
        //    compute update
        div_flux(w1d, 1, udata->dy, dw, *udata);
        //    apply update
        rhodot[IDX(i,j,k,nxl,nyl)] -= dw[0];
        mxdot[ IDX(i,j,k,nxl,nyl)] -= dw[1];
        mydot[ IDX(i,j,k,nxl,nyl)] -= dw[2];
        mzdot[ IDX(i,j,k,nxl,nyl)] -= dw[3];
        Edot[  IDX(i,j,k,nxl,nyl)] -= dw[4];

        // z-directional advection
        //    pack 1D array of variable shortcuts
        pack1D_z(w1d, rho, mx, my, mz, E, udata->Brecv, udata->Frecv, i, j, k, nxl, nyl, nzl);
        //    compute update
        div_flux(w1d, 2, udata->dz, dw, *udata);
        //    apply update
        rhodot[IDX(i,j,k,nxl,nyl)] -= dw[0];
        mxdot[ IDX(i,j,k,nxl,nyl)] -= dw[1];
        mydot[ IDX(i,j,k,nxl,nyl)] -= dw[2];
        mzdot[ IDX(i,j,k,nxl,nyl)] -= dw[3];
        Edot[  IDX(i,j,k,nxl,nyl)] -= dw[4];

      }


  // return with success
  return 0;
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
    opt == 3 means MPI function returns a flag, so check if
             flag != MPI_SUCCESS
    opt == 4 corresponds to a check for an illegal state, so check if
             flag >= 0
*/
int check_flag(void *flagvalue, const string funcname, int opt)
{
  int *errflag;

  // Check if SUNDIALS function returned NULL pointer - no memory allocated
  if (opt == 0 && flagvalue == NULL) {
    cerr << "\nSUNDIALS_ERROR: " << funcname << " failed - returned NULL pointer\n\n";
    return 1; }

  // Check if flag < 0
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      cerr << "\nSUNDIALS_ERROR: " << funcname << " failed with flag = " << *errflag << "\n\n";
      return 1;
    }
  }

  // Check if function returned NULL pointer - no memory allocated
  else if (opt == 2 && flagvalue == NULL) {
    cerr << "\nMEMORY_ERROR: " << funcname << " failed - returned NULL pointer\n\n";
    return 1; }

  // Check if flag != MPI_SUCCESS
  else if (opt == 3) {
    errflag = (int *) flagvalue;
    if (*errflag != MPI_SUCCESS) {
      cerr << "\nMPI_ERROR: " << funcname << " failed with flag = " << *errflag << "\n\n";
      return 1;
    }
  }

  // Check if flag != MPI_SUCCESS
  else if (opt == 4) {
    errflag = (int *) flagvalue;
    if (*errflag != MPI_SUCCESS) {
      cerr << "\nSTATE_ERROR: " << funcname << " failed with flag = " << *errflag;
      if (*errflag == 1)  cerr << "  (illegal density)\n\n";
      if (*errflag == 2)  cerr << "  (illegal energy)\n\n";
      if (*errflag == 3)  cerr << "  (illegal density & energy)\n\n";
      if (*errflag == 4)  cerr << "  (illegal pressure)\n\n";
      if (*errflag == 5)  cerr << "  (illegal density & pressure)\n\n";
      if (*errflag == 6)  cerr << "  (illegal energy & pressure)\n\n";
      if (*errflag == 7)  cerr << "  (illegal density, energy & pressure)\n\n";
      return 1;
    }
  }

  return 0;
}

// Set up parallel decomposition
int UserData::SetupDecomp()
{
  // local variables
  int retval, periods[3], coords[3], nbcoords[3];
  int dims[] = {0, 0, 0};

  // check that this has not been called before
  if (Erecv != NULL || Wrecv != NULL || Srecv != NULL ||
      Nrecv != NULL || Brecv != NULL || Frecv != NULL) {
    cerr << "SetupDecomp warning: parallel decomposition already set up\n";
    return 1;
  }

  // get suggested parallel decomposition
  retval = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (check_flag(&retval, "MPI_Comm_size (UserData::SetupDecomp)", 3)) return -1;
  retval = MPI_Dims_create(nprocs, 3, dims);
  if (check_flag(&retval, "MPI_Dims_create (UserData::SetupDecomp)", 3)) return -1;

  // set up 3D Cartesian communicator
  if ((xlbc*xrbc == 0) && (xlbc+xrbc != 0)) {
    cerr << "SetupDecomp error: only one x-boundary is periodic\n";
    return 1;
  }
  if ((ylbc*yrbc == 0) && (ylbc+yrbc != 0)) {
    cerr << "SetupDecomp error: only one y-boundary is periodic\n";
    return 1;
  }
  if ((zlbc*zrbc == 0) && (zlbc+zrbc != 0)) {
    cerr << "SetupDecomp error: only one z-boundary is periodic\n";
    return 1;
  }
  periods[0] = (xlbc == 0) ? 1 : 0;
  periods[1] = (ylbc == 0) ? 1 : 0;
  periods[2] = (zlbc == 0) ? 1 : 0;
  retval = MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm);
  if (check_flag(&retval, "MPI_Cart_create (UserData::SetupDecomp)", 3)) return -1;
  retval = MPI_Comm_rank(comm, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (UserData::SetupDecomp)", 3)) return -1;

  // determine local extents
  retval = MPI_Cart_get(comm, 3, dims, periods, coords);
  if (check_flag(&retval, "MPI_Cart_get (UserData::SetupDecomp)", 3)) return -1;
  is = nx*(coords[0])/(dims[0]);
  ie = nx*(coords[0]+1)/(dims[0])-1;
  js = ny*(coords[1])/(dims[1]);
  je = ny*(coords[1]+1)/(dims[1])-1;
  ks = nz*(coords[2])/(dims[2]);
  ke = nz*(coords[2]+1)/(dims[2])-1;
  nxl = ie-is+1;
  nyl = je-js+1;
  nzl = ke-ks+1;

  // for all faces where neighbors exist: determine neighbor process indices;
  // for all faces: allocate exchange buffers (external boundaries fill with ghost values)
  Wrecv = new realtype[5*3*nyl*nzl];
  Wsend = new realtype[5*3*nyl*nzl];
  ipW = MPI_PROC_NULL;
  if ((coords[0] > 0) || (xlbc == 0)) {
    nbcoords[0] = coords[0]-1;
    nbcoords[1] = coords[1];
    nbcoords[2] = coords[2];
    retval = MPI_Cart_rank(comm, nbcoords, &ipW);
    if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
  }

  Erecv = new realtype[5*3*nyl*nzl];
  Esend = new realtype[5*3*nyl*nzl];
  ipE = MPI_PROC_NULL;
  if ((coords[0] < dims[0]-1) || (xrbc == 0)) {
    nbcoords[0] = coords[0]+1;
    nbcoords[1] = coords[1];
    nbcoords[2] = coords[2];
    retval = MPI_Cart_rank(comm, nbcoords, &ipE);
    if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
  }

  Srecv = new realtype[5*nxl*3*nzl];
  Ssend = new realtype[5*nxl*3*nzl];
  ipS = MPI_PROC_NULL;
  if ((coords[1] > 0) || (ylbc == 0)) {
    nbcoords[0] = coords[0];
    nbcoords[1] = coords[1]-1;
    nbcoords[2] = coords[2];
    retval = MPI_Cart_rank(comm, nbcoords, &ipS);
    if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
  }

  Nrecv = new realtype[5*nxl*3*nzl];
  Nsend = new realtype[5*nxl*3*nzl];
  ipN = MPI_PROC_NULL;
  if ((coords[1] < dims[1]-1) || (yrbc == 0)) {
    nbcoords[0] = coords[0];
    nbcoords[1] = coords[1]+1;
    nbcoords[2] = coords[2];
    retval = MPI_Cart_rank(comm, nbcoords, &ipN);
    if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
  }

  Brecv = new realtype[5*nxl*nyl*3];
  Bsend = new realtype[5*nxl*nyl*3];
  ipB = MPI_PROC_NULL;
  if ((coords[2] > 0) || (zlbc == 0)) {
    nbcoords[0] = coords[0];
    nbcoords[1] = coords[1];
    nbcoords[2] = coords[2]-1;
    retval = MPI_Cart_rank(comm, nbcoords, &ipB);
    if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
  }

  Frecv = new realtype[5*nxl*nyl*3];
  Fsend = new realtype[5*nxl*nyl*3];
  ipF = MPI_PROC_NULL;
  if ((coords[2] < dims[2]-1) || (zrbc == 0)) {
    nbcoords[0] = coords[0];
    nbcoords[1] = coords[1];
    nbcoords[2] = coords[2]+1;
    retval = MPI_Cart_rank(comm, nbcoords, &ipF);
    if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
  }

  return 0;     // return with success flag
}

// Begin neighbor exchange
int UserData::ExchangeStart(N_Vector w)
{
  // local variables
  int retval, v, i, j, k;

  // access data array
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
  realtype *E = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) E, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;

  // initialize all requests in array
  for (i=0; i<12; i++)  req[i] = MPI_REQUEST_NULL;

  // open an Irecv buffer for each neighbor
  if (ipW != MPI_PROC_NULL) {
    retval = MPI_Irecv(Wrecv, 5*3*nyl*nzl, MPI_SUNREALTYPE, ipW,
                       MPI_ANY_TAG, comm, req);
    if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipE != MPI_PROC_NULL) {
    retval = MPI_Irecv(Erecv, 5*3*nyl*nzl, MPI_SUNREALTYPE, ipE,
                       MPI_ANY_TAG, comm, req+1);
    if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipS != MPI_PROC_NULL) {
    retval = MPI_Irecv(Srecv, 5*nxl*3*nzl, MPI_SUNREALTYPE, ipS,
                       MPI_ANY_TAG, comm, req+2);
  if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipN != MPI_PROC_NULL) {
    retval = MPI_Irecv(Nrecv, 5*nxl*3*nzl, MPI_SUNREALTYPE, ipN,
                       MPI_ANY_TAG, comm, req+3);
    if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipB != MPI_PROC_NULL) {
    retval = MPI_Irecv(Brecv, 5*nxl*nyl*3, MPI_SUNREALTYPE, ipB,
                       MPI_ANY_TAG, comm, req+4);
  if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipF != MPI_PROC_NULL) {
    retval = MPI_Irecv(Frecv, 5*nxl*nyl*3, MPI_SUNREALTYPE, ipF,
                       MPI_ANY_TAG, comm, req+5);
    if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
  }

  // send data to neighbors
  if (ipW != MPI_PROC_NULL) {
    for (k=0; k<nzl; k++)
      for (j=0; j<nyl; j++)
        for (i=0; i<3; i++) {
          Wsend[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(i,j,k,nxl,nyl)];
          Wsend[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[IDX(i,j,k,nxl,nyl)];
          Wsend[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[IDX(i,j,k,nxl,nyl)];
          Wsend[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[IDX(i,j,k,nxl,nyl)];
          Wsend[BUFIDX(4,i,j,k,3,nyl,nzl)] = E[IDX(i,j,k,nxl,nyl)];
        }
    retval = MPI_Isend(Wsend, 5*3*nyl*nzl, MPI_SUNREALTYPE, ipW, 0,
                       comm, req+6);
    if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipE != MPI_PROC_NULL) {
    for (k=0; k<nzl; k++)
      for (j=0; j<nyl; j++)
        for (i=0; i<3; i++) {
          Esend[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(nxl-3+i,j,k,nxl,nyl)];
          Esend[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[IDX(nxl-3+i,j,k,nxl,nyl)];
          Esend[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[IDX(nxl-3+i,j,k,nxl,nyl)];
          Esend[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[IDX(nxl-3+i,j,k,nxl,nyl)];
          Esend[BUFIDX(4,i,j,k,3,nyl,nzl)] = E[IDX(nxl-3+i,j,k,nxl,nyl)];
        }
    retval = MPI_Isend(Esend, 5*3*nyl*nzl, MPI_SUNREALTYPE, ipE, 1,
                       comm, req+7);
    if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipS != MPI_PROC_NULL) {
    for (k=0; k<nzl; k++)
      for (j=0; j<3; j++)
        for (i=0; i<nxl; i++) {
          Ssend[BUFIDX(0,i,j,k,nxl,3,nzl)] = rho[IDX(i,j,k,nxl,nyl)];
          Ssend[BUFIDX(1,i,j,k,nxl,3,nzl)] = mx[IDX(i,j,k,nxl,nyl)];
          Ssend[BUFIDX(2,i,j,k,nxl,3,nzl)] = my[IDX(i,j,k,nxl,nyl)];
          Ssend[BUFIDX(3,i,j,k,nxl,3,nzl)] = mz[IDX(i,j,k,nxl,nyl)];
          Ssend[BUFIDX(4,i,j,k,nxl,3,nzl)] = E[IDX(i,j,k,nxl,nyl)];
        }
    retval = MPI_Isend(Ssend, 5*nxl*3*nzl, MPI_SUNREALTYPE, ipS, 2,
                       comm, req+8);
    if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipN != MPI_PROC_NULL) {
    for (k=0; k<nzl; k++)
      for (j=0; j<3; j++)
        for (i=0; i<nxl; i++) {
          Nsend[BUFIDX(0,i,j,k,nxl,3,nzl)] = rho[IDX(i,nyl-3+j,k,nxl,nyl)];
          Nsend[BUFIDX(1,i,j,k,nxl,3,nzl)] = mx[IDX(i,nyl-3+j,k,nxl,nyl)];
          Nsend[BUFIDX(2,i,j,k,nxl,3,nzl)] = my[IDX(i,nyl-3+j,k,nxl,nyl)];
          Nsend[BUFIDX(3,i,j,k,nxl,3,nzl)] = mz[IDX(i,nyl-3+j,k,nxl,nyl)];
          Nsend[BUFIDX(4,i,j,k,nxl,3,nzl)] = E[IDX(i,nyl-3+j,k,nxl,nyl)];
        }
    retval = MPI_Isend(Nsend, 5*nxl*3*nzl, MPI_SUNREALTYPE, ipN, 3,
                       comm, req+9);
    if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipB != MPI_PROC_NULL) {
    for (k=0; k<3; k++)
      for (j=0; j<nyl; j++)
        for (i=0; i<nxl; i++) {
          Bsend[BUFIDX(0,i,j,k,nxl,nyl,3)] = rho[IDX(i,j,k,nxl,nyl)];
          Bsend[BUFIDX(1,i,j,k,nxl,nyl,3)] = mx[IDX(i,j,k,nxl,nyl)];
          Bsend[BUFIDX(2,i,j,k,nxl,nyl,3)] = my[IDX(i,j,k,nxl,nyl)];
          Bsend[BUFIDX(3,i,j,k,nxl,nyl,3)] = mz[IDX(i,j,k,nxl,nyl)];
          Bsend[BUFIDX(4,i,j,k,nxl,nyl,3)] = E[IDX(i,j,k,nxl,nyl)];
        }
    retval = MPI_Isend(Bsend, 5*nxl*nyl*3, MPI_SUNREALTYPE, ipB, 4,
                       comm, req+10);
    if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
  }

  if (ipF != MPI_PROC_NULL) {
    for (k=0; k<3; k++)
      for (j=0; j<nyl; j++)
        for (i=0; i<nxl; i++) {
          Fsend[BUFIDX(0,i,j,k,nxl,nyl,3)] = rho[IDX(i,j,nzl-3+k,nxl,nyl)];
          Fsend[BUFIDX(1,i,j,k,nxl,nyl,3)] = mx[IDX(i,j,nzl-3+k,nxl,nyl)];
          Fsend[BUFIDX(2,i,j,k,nxl,nyl,3)] = my[IDX(i,j,nzl-3+k,nxl,nyl)];
          Fsend[BUFIDX(3,i,j,k,nxl,nyl,3)] = mz[IDX(i,j,nzl-3+k,nxl,nyl)];
          Fsend[BUFIDX(4,i,j,k,nxl,nyl,3)] = E[IDX(i,j,nzl-3+k,nxl,nyl)];
        }
    retval = MPI_Isend(Fsend, 5*nxl*nyl*3, MPI_SUNREALTYPE, ipF, 5,
                       comm, req+11);
    if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
  }

  // if this process owns any external faces, fill ghost zones
  // to satisfy homogeneous Neumann boundary conditions

  //     West face
  if (ipW == MPI_PROC_NULL) {
    if (xlbc == 1) {  // homogeneous Neumann
      for (k=0; k<nzl; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<3; i++) {
            Wrecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = E[IDX(2-i,j,k,nxl,nyl)];
          }
    } else {          // homogeneous Dirichlet
      for (k=0; k<nzl; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<3; i++) {
            Wrecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = -rho[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = -mx[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = -my[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = -mz[IDX(2-i,j,k,nxl,nyl)];
            Wrecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = -E[IDX(2-i,j,k,nxl,nyl)];
          }
    }
  }

  //     East face
  if (ipE == MPI_PROC_NULL) {
    if (xrbc == 1) {  // homogeneous Neumann
      for (k=0; k<nzl; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<3; i++) {
            Erecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = E[IDX(nxl-3+i,j,k,nxl,nyl)];
          }
    } else {          // homogeneous Dirichlet
      for (k=0; k<nzl; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<3; i++) {
            Erecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = -rho[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = -mx[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = -my[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = -mz[IDX(nxl-3+i,j,k,nxl,nyl)];
            Erecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = -E[IDX(nxl-3+i,j,k,nxl,nyl)];
          }
    }
  }

  //     South face
  if (ipS == MPI_PROC_NULL) {
    if (ylbc == 1) {  // homogeneous Neumann
      for (k=0; k<nzl; k++)
        for (j=0; j<3; j++)
          for (i=0; i<nxl; i++) {
            Srecv[BUFIDX(0,i,j,k,nxl,3,nzl)] = rho[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(1,i,j,k,nxl,3,nzl)] = mx[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(2,i,j,k,nxl,3,nzl)] = my[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(3,i,j,k,nxl,3,nzl)] = mz[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = E[IDX(i,2-j,k,nxl,nyl)];
          }
    } else {          // homogeneous Dirichlet
      for (k=0; k<nzl; k++)
        for (j=0; j<3; j++)
          for (i=0; i<nxl; i++) {
            Srecv[BUFIDX(0,i,j,k,nxl,3,nzl)] = -rho[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(1,i,j,k,nxl,3,nzl)] = -mx[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(2,i,j,k,nxl,3,nzl)] = -my[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(3,i,j,k,nxl,3,nzl)] = -mz[IDX(i,2-j,k,nxl,nyl)];
            Srecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = -E[IDX(i,2-j,k,nxl,nyl)];
          }
    }
  }

  //     North face
  if (ipN == MPI_PROC_NULL) {
    if (yrbc == 1) {  // homogeneous Neumann
      for (k=0; k<nzl; k++)
        for (j=0; j<3; j++)
          for (i=0; i<nxl; i++) {
            Nrecv[BUFIDX(0,i,j,k,nxl,3,nzl)] = rho[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(1,i,j,k,nxl,3,nzl)] = mx[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(2,i,j,k,nxl,3,nzl)] = my[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(3,i,j,k,nxl,3,nzl)] = mz[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = E[IDX(i,nyl-3+j,k,nxl,nyl)];
          }
    } else {          // homogeneous Dirichlet
      for (k=0; k<nzl; k++)
        for (j=0; j<3; j++)
          for (i=0; i<nxl; i++) {
            Nrecv[BUFIDX(0,i,j,k,nxl,3,nzl)] = -rho[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(1,i,j,k,nxl,3,nzl)] = -mx[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(2,i,j,k,nxl,3,nzl)] = -my[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(3,i,j,k,nxl,3,nzl)] = -mz[IDX(i,nyl-3+j,k,nxl,nyl)];
            Nrecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = -E[IDX(i,nyl-3+j,k,nxl,nyl)];
          }
    }
  }

  //     Back face
  if (ipB == MPI_PROC_NULL) {
    if (zlbc == 1) {  // homogeneous Neumann
      for (k=0; k<3; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<nxl; i++) {
            Brecv[BUFIDX(0,i,j,k,nxl,nyl,3)] = rho[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(1,i,j,k,nxl,nyl,3)] = mx[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(2,i,j,k,nxl,nyl,3)] = my[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(3,i,j,k,nxl,nyl,3)] = mz[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = E[IDX(i,j,2-k,nxl,nyl)];
          }
    } else {          // homogeneous Dirichlet
      for (k=0; k<3; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<nxl; i++) {
            Brecv[BUFIDX(0,i,j,k,nxl,nyl,3)] = -rho[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(1,i,j,k,nxl,nyl,3)] = -mx[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(2,i,j,k,nxl,nyl,3)] = -my[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(3,i,j,k,nxl,nyl,3)] = -mz[IDX(i,j,2-k,nxl,nyl)];
            Brecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = -E[IDX(i,j,2-k,nxl,nyl)];
          }
    }
  }

  //     Front face
  if (ipF == MPI_PROC_NULL) {
    if (zrbc == 1) {  // homogeneous Neumann
      for (k=0; k<3; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<nxl; i++) {
            Frecv[BUFIDX(0,i,j,k,nxl,nyl,3)] = rho[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(1,i,j,k,nxl,nyl,3)] = mx[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(2,i,j,k,nxl,nyl,3)] = my[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(3,i,j,k,nxl,nyl,3)] = mz[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = E[IDX(i,j,nzl-3+k,nxl,nyl)];
          }
    } else {          // homogeneous Dirichlet
      for (k=0; k<3; k++)
        for (j=0; j<nyl; j++)
          for (i=0; i<nxl; i++) {
            Frecv[BUFIDX(0,i,j,k,nxl,nyl,3)] = -rho[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(1,i,j,k,nxl,nyl,3)] = -mx[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(2,i,j,k,nxl,nyl,3)] = -my[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(3,i,j,k,nxl,nyl,3)] = -mz[IDX(i,j,nzl-3+k,nxl,nyl)];
            Frecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = -E[IDX(i,j,nzl-3+k,nxl,nyl)];
          }
    }
  }

  return 0;     // return with success flag
}

// Finish neighbor exchange
int UserData::ExchangeEnd()
{
  // local variables
  MPI_Status stat[12];
  int retval;

  // wait for messages to finish send/receive
  retval = MPI_Waitall(12, req, stat);
  if (check_flag(&retval, "MPI_Waitall (UserData::ExchangeEnd)", 3)) return -1;

  return 0;     // return with success flag
}


// Equation of state -- compute and return pressure,
//    p = (gamma-1)*(E - rho/2*(vx^2+vy^2+vz^2), or equivalently
//    p = (gamma-1)*(E - (mx^2+my^2+mz^2)/(2*rho)
inline realtype eos(realtype& rho, realtype& mx, realtype& my,
                    realtype& mz, realtype& E, UserData& udata)
{
  return((udata.gamma-ONE)*(E - (mx*mx+my*my+mz*mz)*HALF/rho));
}

// Check for legal state: returns 0 if density, energy and pressure
// are all positive;  otherwise the return value encodes all failed
// variables:  dfail + efail + pfail, with dfail=0/1, efail=0/2, pfail=0/4,
// i.e., a return of 5 indicates that density and pressure were
// non-positive, but energy was fine
inline int legal_state(realtype& rho, realtype& mx, realtype& my,
                       realtype& mz, realtype& E, UserData& udata)
{
  int dfail, efail, pfail;
  dfail = (rho > ZERO) ? 0 : 1;
  efail = (E > ZERO) ? 0 : 2;
  pfail = (eos(rho, mx, my, mz, E, udata) > ZERO) ? 0 : 4;
  return(dfail+efail+pfail);
}

// Load inputs from file: root process reads parameters and broadcasts results to remaining processes
int load_inputs(int myid, double& xl, double& xr, double& yl,
                double& yr, double& zl, double& zr, double& t0,
                double& tf, double& gamma, long int& nx,
                long int& ny, long int& nz, long int& xlbc,
                long int& xrbc, long int& ylbc, long int& yrbc,
                long int& zlbc, long int& zrbc, long int& showstats)
{
  int retval;
  double dbuff[9];
  long int ibuff[10];
  if (myid == 0) {
    FILE *FID=NULL;
    FID = fopen("input_euler3D.txt","r");
    if (check_flag((void *) FID, "fopen (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"xl = %lf\n", &xl);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"xr = %lf\n", &xr);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"yl = %lf\n", &yl);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"yr = %lf\n", &yr);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"zl = %lf\n", &zl);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"zr = %lf\n", &zr);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"t0 = %lf\n", &t0);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"tf = %lf\n", &tf);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"gamma = %lf\n", &gamma);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"nx = %li\n", &nx);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"ny = %li\n", &ny);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"nz = %li\n", &nz);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"xlbc = %li\n", &xlbc);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"xrbc = %li\n", &xrbc);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"ylbc = %li\n", &ylbc);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"yrbc = %li\n", &yrbc);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"zlbc = %li\n", &zlbc);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"zrbc = %li\n", &zrbc);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    retval = fscanf(FID,"showstats = %li\n", &showstats);
    if (check_flag(&retval, "fscanf (load_inputs)", 0)) return(1);
    fclose(FID);
    ibuff[0] = nx;    // pack buffers
    ibuff[1] = ny;
    ibuff[2] = nz;
    ibuff[3] = xlbc;
    ibuff[4] = xrbc;
    ibuff[5] = ylbc;
    ibuff[6] = yrbc;
    ibuff[7] = zlbc;
    ibuff[8] = zrbc;
    ibuff[9] = showstats;
    dbuff[0] = xl;
    dbuff[1] = xr;
    dbuff[2] = yl;
    dbuff[3] = yr;
    dbuff[4] = zl;
    dbuff[5] = zr;
    dbuff[6] = t0;
    dbuff[7] = tf;
    dbuff[8] = gamma;
  }
  // perform broadcast and unpack results
  retval = MPI_Bcast(dbuff, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (load_inputs)", 3)) return(1);
  retval = MPI_Bcast(ibuff, 10, MPI_LONG, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (load_inputs)", 3)) return(1);
  xl    = dbuff[0];       // unpack buffers
  xr    = dbuff[1];
  yl    = dbuff[2];
  yr    = dbuff[3];
  zl    = dbuff[4];
  zr    = dbuff[5];
  t0    = dbuff[6];
  tf    = dbuff[7];
  gamma = dbuff[8];
  nx    = ibuff[0];
  ny    = ibuff[1];
  nz    = ibuff[2];
  xlbc  = ibuff[3];
  xrbc  = ibuff[4];
  ylbc  = ibuff[5];
  yrbc  = ibuff[6];
  zlbc  = ibuff[7];
  zrbc  = ibuff[8];
  showstats = ibuff[9];

  // return with success
  return(0);
}

// Initial conditions
int initial_conditions(realtype t, N_Vector w, UserData& udata)
{
  // iterate over subdomain, setting initial condition
  long int i, j, k;
  realtype xloc, yloc, zloc;
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *E = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) E, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        xloc = (udata.is+i)*udata.dx + udata.xl;
        yloc = (udata.js+j)*udata.dy + udata.yl;
        zloc = (udata.ks+j)*udata.dz + udata.zl;
        rho[IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        mx[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        my[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        mz[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        E[  IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
      }
  return 0;
}

// External forcing terms
int external_forces(realtype t, N_Vector wdot, UserData& udata)
{
  // iterate over subdomain, applying external forces
  long int i, j, k;
  realtype xloc, yloc, zloc;
  realtype *rhodot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,0);
  if (check_flag((void *) rhodot, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *mxdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,1);
  if (check_flag((void *) mxdot, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *mydot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,2);
  if (check_flag((void *) mydot, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *mzdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,3);
  if (check_flag((void *) mzdot, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  realtype *Edot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,4);
  if (check_flag((void *) Edot, "N_VGetSubvectorArrayPointer (external_forces)", 0)) return -1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        xloc = (udata.is+i)*udata.dx + udata.xl;
        yloc = (udata.js+j)*udata.dy + udata.yl;
        zloc = (udata.ks+j)*udata.dz + udata.zl;
        rhodot[IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        mxdot[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        mydot[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        mzdot[ IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
        Edot[  IDX(i,j,k,udata.nxl,udata.nyl)] = ZERO;
      }
  return 0;
}

// Signed fluxes over patch
void signed_fluxes(realtype (&w1d)[7][5], realtype (&fms)[7][5],
                   realtype (&fps)[7][5], UserData& udata)
{
  // local variables
  realtype rho, mx, my, mz, vx, vy, vz, E, p, csnd, cisq, qsq, h, gamm;
  realtype vr[5], lv[5][5], rv[5][5], lambda[7][5], alpha[5], fx[5], fs[7][5], ws[7][5];
  int i, j, k;

  for (i=0; i<5; i++)
    for (j=0; j<5; j++) {
      rv[i][j] = ZERO;
      lv[i][j] = ZERO;
    }

  // loop over patch, computing relevant quantities at each location
  for (j=0; j<7; j++) {

    // unpack state, and compute corresponding primitive variables
    rho = w1d[j][0];  mx = w1d[j][1];  my = w1d[j][2];  mz = w1d[j][3];  E = w1d[j][4];
    vx = mx/rho;  vy = my/rho;  vz = mz/rho;  p = eos(rho, mx, my, mz, E, udata);

    // compute eigensystem
    csnd = SUNRsqrt(udata.gamma*p/rho);
    cisq = ONE/csnd/csnd;
    qsq = vx*vx + vy*vy + vz*vz;
    gamm = udata.gamma-ONE;
    h = udata.gamma/gamm*p/rho + HALF*qsq;

    lambda[j][0] = abs(vx);
    lambda[j][1] = abs(vx);
    lambda[j][2] = abs(vx);
    lambda[j][3] = abs(vx+csnd);
    lambda[j][4] = abs(vx-csnd);

    rv[0][2] = ONE;
    rv[0][3] = ONE;
    rv[0][4] = ONE;

    rv[1][2] = vx;
    rv[1][3] = vx + csnd;
    rv[1][4] = vx - csnd;

    rv[2][1] = ONE;
    rv[2][2] = vy;
    rv[2][3] = vy;
    rv[2][4] = vy;

    rv[3][0] = ONE;
    rv[3][2] = vz;
    rv[3][3] = vz;
    rv[3][4] = vz;

    rv[4][0] = vz;
    rv[4][1] = vy;
    rv[4][2] = HALF*qsq;
    rv[4][3] = h + vx*csnd;
    rv[4][4] = h - vx*csnd;

    lv[0][0] = -vz;
    lv[0][3] = ONE;

    lv[1][0] = -vy;
    lv[1][2] = ONE;

    lv[2][0] = ONE - HALF*gamm*qsq*cisq;
    lv[2][1] = gamm*vx*cisq;
    lv[2][2] = gamm*vy*cisq;
    lv[2][3] = gamm*vz*cisq;
    lv[2][4] = -gamm*cisq;

    lv[3][0] = FOURTH*gamm*qsq*cisq - HALF*vx/csnd;
    lv[3][1] = HALF/csnd - HALF*gamm*vx*cisq;
    lv[3][2] = -HALF*gamm*vy*cisq;
    lv[3][3] = -HALF*gamm*vz*cisq;
    lv[3][4] = HALF*gamm*cisq;

    lv[4][0] = FOURTH*gamm*qsq*cisq + HALF*vx/csnd;
    lv[4][1] = -HALF/csnd - HALF*gamm*vx*cisq;
    lv[4][2] = -HALF*gamm*vy*cisq;
    lv[4][3] = -HALF*gamm*vz*cisq;
    lv[4][4] = HALF*gamm*cisq;

    // compute basic flux
    fx[0] = mx;
    fx[1] = rho*vx*vx + p;
    fx[2] = rho*vx*vy;
    fx[3] = rho*vx*vz;
    fx[4] = vx*(E + p);

    // compute projected flux stencils
    for (i=0; i<5; i++)
      fs[j][i] = lv[j][0]*fx[0] + lv[j][1]*fx[1] + lv[j][2]*fx[2] + lv[j][3]*fx[3] + lv[j][4]*fx[4];
    for (i=0; i<5; i++)
      ws[j][i] = lv[j][0]*rho + lv[j][1]*mx + lv[j][2]*my + lv[j][3]*mz + lv[j][4]*E;

  }

  // accumulate "alpha" values for patch
  for (j=0; j<5; j++) {
    alpha[j] = lambda[0][j];
    for (i=1; i<7; i++)
      alpha[j] = max(alpha[j], lambda[i][j]);
  }

  // compute signed fluxes over patch: fms is left-shifted, fps is right-shifted
  for (j=0; j<7; j++)
    for (i=0; i<5; i++) {
      fms[j][i] = HALF*(fs[j][i] - alpha[i]*ws[j][i]);
      fps[j][i] = HALF*(fs[j][i] + alpha[i]*ws[j][i]);
    }

}

// Utility routine to print solution statistics
//    firstlast = 0 indicates the first output
//    firstlast = 1 indicates a normal output
//    firstlast = 2 indicates the lastoutput
int print_stats(realtype t, N_Vector w, int firstlast, UserData& udata)
{
  realtype rmsvals[] = {ZERO, ZERO, ZERO, ZERO, ZERO};
  bool outproc = (udata.myid == 0);
  long int v, i, j, k;
  int retval;
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *E = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) E, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  if (firstlast < 2) {
    for (k=0; k<udata.nzl; k++)
      for (j=0; j<udata.nyl; j++)
        for (i=0; i<udata.nxl; i++) {
          rmsvals[0] += SUNRpowerI(rho[IDX(i,j,k,udata.nxl,udata.nyl)], 2);
          rmsvals[1] += SUNRpowerI( mx[IDX(i,j,k,udata.nxl,udata.nyl)], 2);
          rmsvals[2] += SUNRpowerI( my[IDX(i,j,k,udata.nxl,udata.nyl)], 2);
          rmsvals[3] += SUNRpowerI( mz[IDX(i,j,k,udata.nxl,udata.nyl)], 2);
          rmsvals[4] += SUNRpowerI(  E[IDX(i,j,k,udata.nxl,udata.nyl)], 2);
        }
    retval = MPI_Reduce(MPI_IN_PLACE, &rmsvals, 5, MPI_SUNREALTYPE, MPI_SUM, 0, udata.comm);
    if (check_flag(&retval, "MPI_Reduce (print_stats)", 3)) MPI_Abort(udata.comm, 1);
    for (v=0; v<5; v++)  rmsvals[v] = SUNRsqrt(rmsvals[v]/udata.nx/udata.ny/udata.nz);
  }
  if (!outproc)  return(0);
  if (firstlast == 0)
    cout << "\n        t      ||rho||_rms   ||mx||_rms   ||my||_rms   ||mz||_rms   ||E||_rms\n";
  if (firstlast != 1)
    cout << "   --------------------------------------------------------------------------------\n";
  if (firstlast<2)
    printf("  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n", t,
           rmsvals[0], rmsvals[1], rmsvals[2], rmsvals[3], rmsvals[4]);
  return(0);
}

// Utility routine to output the current solution
//    newappend == 1 indicates create a new file
//    newappend == 0 indicates append to existing file
int output_solution(N_Vector w, int newappend, UserData& udata)
{
  // reusable variables
  char outtype[2];
  char outname[100];
  FILE *FID;
  realtype *W;
  long int i;
  long int N = udata.nzl * udata.nyl * udata.nxl;

  // Set string for output type
  if (newappend == 1) {
    sprintf(outtype, "w");
  } else {
    sprintf(outtype, "a");
  }

  // Output density
  sprintf(outname, "euler3d_rho.%03i.txt", udata.myid);  // filename
  FID = fopen(outname,outtype);                          // file ptr
  W = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);    // data array
  if (check_flag((void *) W, "N_VGetSubvectorArrayPointer (output_solution)", 0)) return -1;
  for (i=0; i<N; i++) fprintf(FID," %.16e", W[i]);       // output
  fprintf(FID,"\n");                                     // newline
  fclose(FID);                                           // close file

  // Output x-momentum
  sprintf(outname, "euler3d_mx.%03i.txt", udata.myid);
  FID = fopen(outname,outtype);
  W = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) W, "N_VGetSubvectorArrayPointer (output_solution)", 0)) return -1;
  for (i=0; i<N; i++) fprintf(FID," %.16e", W[i]);
  fprintf(FID,"\n");
  fclose(FID);

  // Output y-momentum
  sprintf(outname, "euler3d_my.%03i.txt", udata.myid);
  FID = fopen(outname,outtype);
  W = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) W, "N_VGetSubvectorArrayPointer (output_solution)", 0)) return -1;
  for (i=0; i<N; i++) fprintf(FID," %.16e", W[i]);
  fprintf(FID,"\n");
  fclose(FID);

  // Output z-momentum
  sprintf(outname, "euler3d_mz.%03i.txt", udata.myid);
  FID = fopen(outname,outtype);
  W = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) W, "N_VGetSubvectorArrayPointer (output_solution)", 0)) return -1;
  for (i=0; i<N; i++) fprintf(FID," %.16e", W[i]);
  fprintf(FID,"\n");
  fclose(FID);

  // Output energy
  sprintf(outname, "euler3d_e.%03i.txt", udata.myid);
  FID = fopen(outname,outtype);
  W = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) W, "N_VGetSubvectorArrayPointer (output_solution)", 0)) return -1;
  for (i=0; i<N; i++) fprintf(FID," %.16e", W[i]);
  fprintf(FID,"\n");
  fclose(FID);

  // return with success
  return(0);
}

// Utility routines to pack 1-dimensional data, properly handling
// solution vs receive buffers
inline void pack1D_x(realtype (&w1d)[7][5], realtype* rho, realtype* mx,
                     realtype* my, realtype* mz, realtype* E,
                     realtype* Wrecv, realtype* Erecv,
                     long int& i, long int& j, long int& k,
                     long int& nxl, long int& nyl, long int& nzl)
{
  for (int l=0; l<4; l++) {
    w1d[l][0] = (i<(3-l)) ? Wrecv[BUFIDX(0,i+l,j,k,3,nyl,nzl)] : rho[IDX(i-3+l,j,k,nxl,nyl)];
    w1d[l][1] = (i<(3-l)) ? Wrecv[BUFIDX(1,i+l,j,k,3,nyl,nzl)] : mx[IDX(i-3+l,j,k,nxl,nyl)];
    w1d[l][2] = (i<(3-l)) ? Wrecv[BUFIDX(2,i+l,j,k,3,nyl,nzl)] : my[IDX(i-3+l,j,k,nxl,nyl)];
    w1d[l][3] = (i<(3-l)) ? Wrecv[BUFIDX(3,i+l,j,k,3,nyl,nzl)] : mz[IDX(i-3+l,j,k,nxl,nyl)];
    w1d[l][4] = (i<(3-l)) ? Wrecv[BUFIDX(4,i+l,j,k,3,nyl,nzl)] : E[IDX(i-3+l,j,k,nxl,nyl)];
  }
  w1d[3][0] = rho[IDX(i,j,k,nxl,nyl)];
  w1d[3][1] = mx[IDX(i,j,k,nxl,nyl)];
  w1d[3][2] = my[IDX(i,j,k,nxl,nyl)];
  w1d[3][3] = mz[IDX(i,j,k,nxl,nyl)];
  w1d[3][4] = E[IDX(i,j,k,nxl,nyl)];
  for (int l=1; l<4; l++) {
    w1d[l+3][0] = (i>(nxl-l-1)) ? Erecv[BUFIDX(0,i-nxl+l,j,k,3,nyl,nzl)] : rho[IDX(i+l,j,k,nxl,nyl)];
    w1d[l+3][1] = (i>(nxl-l-1)) ? Erecv[BUFIDX(1,i-nxl+l,j,k,3,nyl,nzl)] : mx[IDX(i+l,j,k,nxl,nyl)];
    w1d[l+3][2] = (i>(nxl-l-1)) ? Erecv[BUFIDX(2,i-nxl+l,j,k,3,nyl,nzl)] : my[IDX(i+l,j,k,nxl,nyl)];
    w1d[l+3][3] = (i>(nxl-l-1)) ? Erecv[BUFIDX(3,i-nxl+l,j,k,3,nyl,nzl)] : mz[IDX(i+l,j,k,nxl,nyl)];
    w1d[l+3][4] = (i>(nxl-l-1)) ? Erecv[BUFIDX(4,i-nxl+l,j,k,3,nyl,nzl)] : E[IDX(i+l,j,k,nxl,nyl)];
  }
}
inline void pack1D_y(realtype (&w1d)[7][5], realtype* rho, realtype* mx,
                     realtype* my, realtype* mz, realtype* E,
                     realtype* Srecv, realtype* Nrecv,
                     long int& i, long int& j,  long int& k,
                     long int& nxl, long int& nyl, long int& nzl)
{
  for (int l=0; l<4; l++) {
      w1d[l][0] = (j<(3-l)) ? Srecv[BUFIDX(0,i,j+l,k,nxl,3,nzl)] : rho[IDX(i,j-3+l,k,nxl,nyl)];
      w1d[l][1] = (j<(3-l)) ? Srecv[BUFIDX(1,i,j+l,k,nxl,3,nzl)] : mx[IDX(i,j-3+l,k,nxl,nyl)];
      w1d[l][2] = (j<(3-l)) ? Srecv[BUFIDX(2,i,j+l,k,nxl,3,nzl)] : my[IDX(i,j-3+l,k,nxl,nyl)];
      w1d[l][3] = (j<(3-l)) ? Srecv[BUFIDX(3,i,j+l,k,nxl,3,nzl)] : mz[IDX(i,j-3+l,k,nxl,nyl)];
      w1d[l][4] = (j<(3-l)) ? Srecv[BUFIDX(4,i,j+l,k,nxl,3,nzl)] : E[IDX(i,j-3+l,k,nxl,nyl)];
  }
  w1d[3][0] = rho[IDX(i,j,k,nxl,nyl)];
  w1d[3][1] = mx[IDX(i,j,k,nxl,nyl)];
  w1d[3][2] = my[IDX(i,j,k,nxl,nyl)];
  w1d[3][3] = mz[IDX(i,j,k,nxl,nyl)];
  w1d[3][4] = E[IDX(i,j,k,nxl,nyl)];
  for (int l=1; l<4; l++) {
    w1d[l+3][0] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(0,i,j-nyl+l,k,nxl,3,nzl)] : rho[IDX(i,j+l,k,nxl,nyl)];
    w1d[l+3][1] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(1,i,j-nyl+l,k,nxl,3,nzl)] : mx[IDX(i,j+l,k,nxl,nyl)];
    w1d[l+3][2] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(2,i,j-nyl+l,k,nxl,3,nzl)] : my[IDX(i,j+l,k,nxl,nyl)];
    w1d[l+3][3] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(3,i,j-nyl+l,k,nxl,3,nzl)] : mz[IDX(i,j+l,k,nxl,nyl)];
    w1d[l+3][4] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(4,i,j-nyl+l,k,nxl,3,nzl)] : E[IDX(i,j+l,k,nxl,nyl)];
  }
}
inline void pack1D_z(realtype (&w1d)[7][5], realtype* rho, realtype* mx,
                     realtype* my, realtype* mz, realtype* E,
                     realtype* Brecv, realtype* Frecv,
                     long int& i, long int& j, long int& k,
                     long int& nxl, long int& nyl, long int& nzl)
{
  for (int l=0; l<4; l++) {
    w1d[l][0] = (k<(3-l)) ? Brecv[BUFIDX(0,i,j,k+l,nxl,nyl,3)] : rho[IDX(i,j,k-3+l,nxl,nyl)];
    w1d[l][1] = (k<(3-l)) ? Brecv[BUFIDX(1,i,j,k+l,nxl,nyl,3)] : mx[IDX(i,j,k-3+l,nxl,nyl)];
    w1d[l][2] = (k<(3-l)) ? Brecv[BUFIDX(2,i,j,k+l,nxl,nyl,3)] : my[IDX(i,j,k-3+l,nxl,nyl)];
    w1d[l][3] = (k<(3-l)) ? Brecv[BUFIDX(3,i,j,k+l,nxl,nyl,3)] : mz[IDX(i,j,k-3+l,nxl,nyl)];
    w1d[l][4] = (k<(3-l)) ? Brecv[BUFIDX(4,i,j,k+l,nxl,nyl,3)] : E[IDX(i,j,k-3+l,nxl,nyl)];
  }
  w1d[3][0] = rho[IDX(i,j,k,nxl,nyl)];
  w1d[3][1] = mx[IDX(i,j,k,nxl,nyl)];
  w1d[3][2] = my[IDX(i,j,k,nxl,nyl)];
  w1d[3][3] = mz[IDX(i,j,k,nxl,nyl)];
  w1d[3][4] = E[IDX(i,j,k,nxl,nyl)];
  for (int l=1; l<4; l++) {
    w1d[l+3][0] = (k>(nzl-l-1)) ? Frecv[BUFIDX(0,i,j,k-nzl+l,nxl,nyl,3)] : rho[IDX(i,j,k+l,nxl,nyl)];
    w1d[l+3][1] = (k>(nzl-l-1)) ? Frecv[BUFIDX(1,i,j,k-nzl+l,nxl,nyl,3)] : mx[IDX(i,j,k+l,nxl,nyl)];
    w1d[l+3][2] = (k>(nzl-l-1)) ? Frecv[BUFIDX(2,i,j,k-nzl+l,nxl,nyl,3)] : my[IDX(i,j,k+l,nxl,nyl)];
    w1d[l+3][3] = (k>(nzl-l-1)) ? Frecv[BUFIDX(3,i,j,k-nzl+l,nxl,nyl,3)] : mz[IDX(i,j,k+l,nxl,nyl)];
    w1d[l+3][4] = (k>(nzl-l-1)) ? Frecv[BUFIDX(4,i,j,k-nzl+l,nxl,nyl,3)] : E[IDX(i,j,k+l,nxl,nyl)];
  }
}

// given a 7-point stencil of solution values, advection
// coefficient and spatial mesh increment, compute Div(flux)
// at the center node (store in dw)
//    idir = 0  implies x-directional flux
//    idir = 1  implies y-directional flux
//    idir = 2  implies z-directional flux
void div_flux(realtype (&w1d)[7][5], int idir, realtype& dx, realtype* dw, UserData& udata)
{
  // local data
  int i, j;
  realtype fs[7][5], ws[7][5], alpha[7][5], fps[7][5], fms[7][5], eta,
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11,
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;
  const realtype d0 = RCONST(0.3);
  const realtype d1 = RCONST(0.6);
  const realtype d2 = RCONST(0.1);
  const realtype d3 = RCONST(0.1);
  const realtype d4 = RCONST(0.6);
  const realtype d5 = RCONST(0.3);
  const realtype bc = RCONST(13.0)/RCONST(12.0);
  const realtype sixth = RCONST(1.0)/RCONST(6.0);
  const realtype epsilon = 1e-6;

  // convert state to direction-independent version
  if (idir > 0)
    for (i=0; i<7; i++) swap(w1d[i][1], w1d[i][1+idir]);

  // compute signed fluxes over patch
  signed_fluxes(w1d, fms, fps, udata);

  // perform WENO flux calculation based on shifted fluxes
  for (j=0; j<5; j++) {
    p0  = (      -fms[2][j] +  FIVE*fms[3][j] +    TWO*fms[4][j])*sixth;
    p1  = (   TWO*fms[3][j] +  FIVE*fms[4][j] -        fms[5][j])*sixth;
    p2  = (ELEVEN*fms[4][j] - SEVEN*fms[5][j] +    TWO*fms[6][j])*sixth;
    p3  = (      -fms[1][j] +  FIVE*fms[2][j] +    TWO*fms[3][j])*sixth;
    p4  = (   TWO*fms[2][j] +  FIVE*fms[3][j] -        fms[4][j])*sixth;
    p5  = (ELEVEN*fms[3][j] - SEVEN*fms[4][j] +    TWO*fms[5][j])*sixth;
    p6  = (   TWO*fps[1][j] - SEVEN*fps[2][j] + ELEVEN*fps[3][j])*sixth;
    p7  = (      -fps[2][j] +  FIVE*fps[3][j] +    TWO*fps[4][j])*sixth;
    p8  = (   TWO*fps[3][j] +  FIVE*fps[4][j] -        fps[5][j])*sixth;
    p9  = (   TWO*fps[0][j] - SEVEN*fps[1][j] + ELEVEN*fps[2][j])*sixth;
    p10 = (      -fps[1][j] +  FIVE*fps[2][j] +    TWO*fps[3][j])*sixth;
    p11 = (   TWO*fps[2][j] +  FIVE*fps[3][j] -        fps[4][j])*sixth;

    b0  = bc*pow(        fms[2][j] -  TWO*fms[3][j] +       fms[4][j],2)
      + FOURTH*pow(      fms[2][j] - FOUR*fms[3][j] + THREE*fms[4][j],2);
    b1  = bc*pow(        fms[3][j] -  TWO*fms[4][j] +       fms[5][j],2)
      + FOURTH*pow(      fms[3][j] -      fms[5][j],2);
    b2  = bc*pow(        fms[4][j] -  TWO*fms[5][j] +       fms[6][j],2)
      + FOURTH*pow(THREE*fms[4][j] - FOUR*fms[5][j] +       fms[6][j],2);
    b3  = bc*pow(        fms[1][j] -  TWO*fms[2][j] +       fms[3][j],2)
      + FOURTH*pow(      fms[1][j] - FOUR*fms[2][j] + THREE*fms[3][j],2);
    b4  = bc*pow(        fms[2][j] -  TWO*fms[3][j] +       fms[4][j],2)
      + FOURTH*pow(      fms[2][j] -      fms[4][j],2);
    b5  = bc*pow(        fms[3][j] -  TWO*fms[4][j] +       fms[5][j],2)
      + FOURTH*pow(THREE*fms[3][j] - FOUR*fms[4][j] +       fms[5][j],2);
    b6  = bc*pow(        fps[1][j] -  TWO*fps[2][j] +       fps[3][j],2)
      + FOURTH*pow(      fps[1][j] - FOUR*fps[2][j] + THREE*fps[3][j],2);
    b7  = bc*pow(        fps[2][j] -  TWO*fps[3][j] +       fps[4][j],2)
      + FOURTH*pow(      fps[2][j] -      fps[4][j],2);
    b8  = bc*pow(        fps[3][j] -  TWO*fps[4][j] +       fps[5][j],2)
      + FOURTH*pow(THREE*fps[3][j] - FOUR*fps[4][j] +       fps[5][j],2);
    b9  = bc*pow(        fps[0][j] -  TWO*fps[1][j] +       fps[2][j],2)
      + FOURTH*pow(      fps[0][j] - FOUR*fps[1][j] + THREE*fps[2][j],2);
    b10 = bc*pow(        fps[1][j] -  TWO*fps[2][j] +       fps[3][j],2)
      + FOURTH*pow(      fps[1][j] -      fps[3][j],2);
    b11 = bc*pow(        fps[2][j] -  TWO*fps[3][j] +       fps[4][j],2)
      + FOURTH*pow(THREE*fps[2][j] - FOUR*fps[3][j] +       fps[4][j],2);

    a0  = d0 / (epsilon + b0)  / (epsilon + b0);
    a1  = d1 / (epsilon + b1)  / (epsilon + b1);
    a2  = d2 / (epsilon + b2)  / (epsilon + b2);
    a3  = d0 / (epsilon + b3)  / (epsilon + b3);
    a4  = d1 / (epsilon + b4)  / (epsilon + b4);
    a5  = d2 / (epsilon + b5)  / (epsilon + b5);
    a6  = d3 / (epsilon + b6)  / (epsilon + b6);
    a7  = d4 / (epsilon + b7)  / (epsilon + b7);
    a8  = d5 / (epsilon + b8)  / (epsilon + b8);
    a9  = d3 / (epsilon + b9)  / (epsilon + b9);
    a10 = d4 / (epsilon + b10) / (epsilon + b10);
    a11 = d5 / (epsilon + b11) / (epsilon + b11);

    dw[j] = ( (a0*p0 + a1*p1 + a2*p2)/(a0 + a1 + a2) -
              (a3*p3 + a4*p4 + a5*p5)/(a3 + a4 + a5) +
              (a6*p6 + a7*p7 + a8*p8)/(a6 + a7 + a8) -
              (a9*p9 + a10*p10 + a11*p11)/(a9 + a10 + a11) ) / dx;
  }

  // convert flux to direction-independent version
  if (idir > 0) swap(dw[1], dw[1+idir]);

}

//---- end of file ----
