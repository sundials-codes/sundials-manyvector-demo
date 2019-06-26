/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Implementation file for shared main routine and utility routines.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>

#ifdef DEBUG
#include "fenv.h"
#endif

// prototypes of local functions to be provided to ARKode
//    f routine to compute the ODE RHS function f(t,y).
static int f(realtype t, N_Vector w, N_Vector wdot, void* user_data);
//    stability routine to compute maximum stable step size
static int stability(N_Vector w, realtype t, realtype* dt_stab, void* user_data);


// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem parameters
  long int N, Ntot, i;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int dense_order;               // dense output order of accuracy
  int idense;                    // flag denoting integration type (dense output vs tstop)
  int imex;                      // flag denoting class of method (0=implicit, 1=explicit, 2=IMEX)
  int fixedpt;                   // flag denoting use of fixed-point nonlinear solver
  int myid;                      // MPI process ID
  N_Vector w = NULL;             // empty vectors for storing overall solution
  N_Vector wsub[NVAR];
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
  retval = load_inputs(myid, argc, argv, udata, opts);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (retval > 0) MPI_Abort(MPI_COMM_WORLD, 0);
  realtype dTout = (udata.tf-udata.t0)/(udata.nout);

  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);

  // Initial problem output
  bool outproc = (udata.myid == 0);
  if (outproc) {
    cout << "\n3D compressible inviscid Euler test problem:\n";
    cout << "   nprocs: " << udata.nprocs << " (" << udata.npx << " x "
         << udata.npy << " x " << udata.npz << ")\n";
    cout << "   spatial domain: [" << udata.xl << ", " << udata.xr << "] x ["
         << udata.yl << ", " << udata.yr << "] x ["
         << udata.zl << ", " << udata.zr << "]\n";
    cout << "   time domain = (" << udata.t0 << ", " << udata.tf << "]\n";
    cout << "   bdry cond (0=per, 1=Neu, 2=Dir): ["
         << udata.xlbc << ", " << udata.xrbc << "] x ["
         << udata.ylbc << ", " << udata.yrbc << "] x ["
         << udata.zlbc << ", " << udata.zrbc << "]\n";
    cout << "   gamma: " << udata.gamma << "\n";
    cout << "   cfl fraction: " << udata.cfl << "\n";
    cout << "   spatial grid: " << udata.nx << " x " << udata.ny << " x "
         << udata.nz << "\n";
  }
  if (udata.showstats)
    printf("      proc %4i: %li x %li x %li\n", udata.myid, udata.nxl, udata.nyl, udata.nzl);
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);

  // open solver diagnostics output file for writing
  FILE *DFID = NULL;
  if (outproc)
    DFID=fopen("diags_euler3D.txt","w");

  // Initialize N_Vector data structures
  N = (udata.nxl)*(udata.nyl)*(udata.nzl);
  Ntot = (udata.nx)*(udata.ny)*(udata.nz);
  for (i=0; i<NVAR; i++) {
    wsub[i] = NULL;
    wsub[i] = N_VNew_Parallel(udata.comm, N, Ntot);
    if (check_flag((void *) wsub[i], "N_VNew_Parallel (main)", 0)) MPI_Abort(udata.comm, 1);
  }
  w = N_VNew_MPIManyVector(NVAR, wsub);  // combined solution vector
  if (check_flag((void *) w, "N_VNew_MPIManyVector (main)", 0)) MPI_Abort(udata.comm, 1);

  // set initial conditions
  retval = initial_conditions(udata.t0, w, udata);         // Set initial conditions
  if (check_flag(&retval, "initial_conditions (main)", 1)) MPI_Abort(udata.comm, 1);

  // initialize the integrator memory  
  arkode_mem = ARKStepCreate(f, NULL, udata.t0, w);
  if (check_flag(arkode_mem, "ARKStepCreate (main)", 1)) MPI_Abort(udata.comm, 1);
  
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
  
  // Each processor outputs domain/subdomain information
  char outname[100];
  sprintf(outname, "output-euler3D_subdomain.%07i.txt", udata.myid);
  FILE *UFID = fopen(outname,"w");
  fprintf(UFID, "%li  %li  %li  %li  %li  %li  %li  %li  %li  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf\n",
	  udata.nx, udata.ny, udata.nz, udata.is, udata.ie,
          udata.js, udata.je, udata.ks, udata.ke, udata.xl, udata.xr,
          udata.yl, udata.yr, udata.zl, udata.zr, udata.t0, udata.tf, dTout);
  fclose(UFID);

  // Output initial conditions to disk
  retval = output_solution(w, 1, udata);
  if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);

  // output diagnostic information (if applicable)
  retval = output_diagnostics(udata.t0, w, udata);
  if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

  // compute setup time
  double tsetup = MPI_Wtime() - tstart;
  tstart = MPI_Wtime();

  // If (dense_order == -1), use tstop mode
  if (opts.dense_order == -1)
    idense = 0;
  else   // otherwise tell integrator to use dense output
    idense = 1;

  /* Main time-stepping loop: calls ARKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  realtype t = udata.t0;
  realtype tout = udata.t0+dTout;
  if (udata.showstats) {
    retval = print_stats(t, w, 0, udata);
    if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  int iout;
  for (iout=0; iout<udata.nout; iout++) {

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

    // output statistics to stdout
    if (udata.showstats) {
      retval = print_stats(t, w, 1, udata);
      if (check_flag(&retval, "print_stats (main)", 1)) MPI_Abort(udata.comm, 1);
    }

    // output diagnostic information (if applicable)
    retval = output_diagnostics(t, w, udata);
    if (check_flag(&retval, "output_diagnostics (main)", 1)) MPI_Abort(udata.comm, 1);

    // output results to disk
    retval = output_solution(w, 0, udata);
    if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
  }
  if (udata.showstats) {
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
  for (i=0; i<NVAR; i++)
    N_VDestroy(wsub[i]);
  ARKStepFree(&arkode_mem);    // Free integrator memory
  MPI_Finalize();              // Finalize MPI
  return 0;
}


//-----------------------------------------------------------
// Functions called by ARKode
//-----------------------------------------------------------

// f routine to compute the ODE RHS function f(t,y).
static int f(realtype t, N_Vector w, N_Vector wdot, void *user_data)
{
  // access problem data
  UserData *udata = (UserData *) user_data;

  // initialize output to zeros
  N_VConst(ZERO, wdot);

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *rhodot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,0);
  if (check_flag((void *) rhodot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mxdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,1);
  if (check_flag((void *) mxdot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mydot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,2);
  if (check_flag((void *) mydot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *mzdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,3);
  if (check_flag((void *) mzdot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;
  realtype *etdot = N_VGetSubvectorArrayPointer_MPIManyVector(wdot,4);
  if (check_flag((void *) etdot, "N_VGetSubvectorArrayPointer (f)", 0)) return -1;

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
  realtype w1d[6][NVAR];
  long int v, i, j, k;

  // compute face-centered fluxes over subdomain interior
  for (k=3; k<nzl-2; k++)
    for (j=3; j<nyl-2; j++)
      for (i=3; i<nxl-2; i++) {

        // return with failure on non-positive density, energy or pressure
        // (only check this first time)
        retval = udata->legal_state(rho[IDX(i,j,k,nxl,nyl,nzl)], mx[IDX(i,j,k,nxl,nyl,nzl)],
                                    my[ IDX(i,j,k,nxl,nyl,nzl)], mz[IDX(i,j,k,nxl,nyl,nzl)],
                                    et[ IDX(i,j,k,nxl,nyl,nzl)]);
        if (check_flag(&retval, "legal_state (f)", 1)) return -1;

        // pack 1D x-directional array of variable shortcuts
        udata->pack1D_x(w1d, rho, mx, my, mz, et, i, j, k);
        // compute flux at lower x-directional face
        face_flux(w1d, 0, &(udata->xflux[BUFIDX(0,i,j,k,nxl+1,nyl,nzl)]), *udata);

        // pack 1D y-directional array of variable shortcuts
        udata->pack1D_y(w1d, rho, mx, my, mz, et, i, j, k);
        // compute flux at lower y-directional face
        face_flux(w1d, 1, &(udata->yflux[BUFIDX(0,i,j,k,nxl,nyl+1,nzl)]), *udata);

        // pack 1D z-directional array of variable shortcuts
        udata->pack1D_z(w1d, rho, mx, my, mz, et, i, j, k);
        // compute flux at lower z-directional face
        face_flux(w1d, 2, &(udata->zflux[BUFIDX(0,i,j,k,nxl,nyl,nzl+1)]), *udata);

      }

  // wait for boundary data to arrive from neighbors
  retval = udata->ExchangeEnd();
  if (check_flag(&retval, "ExchangeEnd (f)", 1)) return -1;

  // computing remaining fluxes over boundary (loop over entire domain, but skip interior)
  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {

        // skip strict interior (already computed)
        if ( (k>2) && (k<nzl-2) && (j>2) && (j<nyl-2) && (i>2) && (i<nxl-2) ) continue;

        // return with failure on non-positive density, energy or pressure
        retval = udata->legal_state(rho[IDX(i,j,k,nxl,nyl,nzl)], mx[IDX(i,j,k,nxl,nyl,nzl)],
                                    my[ IDX(i,j,k,nxl,nyl,nzl)], mz[IDX(i,j,k,nxl,nyl,nzl)],
                                    et[ IDX(i,j,k,nxl,nyl,nzl)]);
        if (check_flag(&retval, "legal_state (f)", 1)) return -1;

        // x-directional fluxes at "lower" face
        udata->pack1D_x_bdry(w1d, rho, mx, my, mz, et, i, j, k);
        face_flux(w1d, 0, &(udata->xflux[BUFIDX(0,i,j,k,nxl+1,nyl,nzl)]), *udata);

        // x-directional fluxes at "upper" boundary face
        if (i == nxl-1) {
          udata->pack1D_x_bdry(w1d, rho, mx, my, mz, et, nxl, j, k);
          face_flux(w1d, 0, &(udata->xflux[BUFIDX(0,nxl,j,k,nxl+1,nyl,nzl)]), *udata);
        }

        // y-directional fluxes at "lower" face
        udata->pack1D_y_bdry(w1d, rho, mx, my, mz, et, i, j, k);
        face_flux(w1d, 1, &(udata->yflux[BUFIDX(0,i,j,k,nxl,nyl+1,nzl)]), *udata);

        // y-directional fluxes at "upper" boundary face
        if (j == nyl-1) {
          udata->pack1D_y_bdry(w1d, rho, mx, my, mz, et, i, nyl, k);
          face_flux(w1d, 1, &(udata->yflux[BUFIDX(0,i,nyl,k,nxl,nyl+1,nzl)]), *udata);
        }

        // z-directional fluxes at "lower" face
        udata->pack1D_z_bdry(w1d, rho, mx, my, mz, et, i, j, k);
        face_flux(w1d, 2, &(udata->zflux[BUFIDX(0,i,j,k,nxl,nyl,nzl+1)]), *udata);

        // z-directional fluxes at "upper" boundary face
        if (k == nzl-1) {
          udata->pack1D_z_bdry(w1d, rho, mx, my, mz, et, i, j, nzl);
          face_flux(w1d, 2, &(udata->zflux[BUFIDX(0,i,j,nzl,nxl,nyl,nzl+1)]), *udata);
        }

      }

  // iterate over subdomain, updating RHS (source terms were already included)
  for (k=0; k<nzl; k++)
    for (j=0; j<nyl; j++)
      for (i=0; i<nxl; i++) {
        rhodot[IDX(i,j,k,nxl,nyl,nzl)] -= ( ( udata->xflux[BUFIDX(0,i+1,j,  k,  nxl+1,nyl,  nzl  )]
                                            - udata->xflux[BUFIDX(0,i,  j,  k,  nxl+1,nyl,  nzl  )])/(udata->dx)
                                          + ( udata->yflux[BUFIDX(0,i,  j+1,k,  nxl,  nyl+1,nzl  )]
                                            - udata->yflux[BUFIDX(0,i,  j,  k,  nxl,  nyl+1,nzl  )])/(udata->dy)
                                          + ( udata->zflux[BUFIDX(0,i,  j,  k+1,nxl,  nyl,  nzl+1)]
                                            - udata->zflux[BUFIDX(0,i,  j,  k,  nxl,  nyl,  nzl+1)])/(udata->dz) );
        mxdot[ IDX(i,j,k,nxl,nyl,nzl)] -= ( ( udata->xflux[BUFIDX(1,i+1,j,  k,  nxl+1,nyl,  nzl  )]
                                            - udata->xflux[BUFIDX(1,i,  j,  k,  nxl+1,nyl,  nzl  )])/(udata->dx)
                                          + ( udata->yflux[BUFIDX(1,i,  j+1,k,  nxl,  nyl+1,nzl  )]
                                            - udata->yflux[BUFIDX(1,i,  j,  k,  nxl,  nyl+1,nzl  )])/(udata->dy)
                                          + ( udata->zflux[BUFIDX(1,i,  j,  k+1,nxl,  nyl,  nzl+1)]
                                            - udata->zflux[BUFIDX(1,i,  j,  k,  nxl,  nyl,  nzl+1)])/(udata->dz) );
        mydot[ IDX(i,j,k,nxl,nyl,nzl)] -= ( ( udata->xflux[BUFIDX(2,i+1,j,  k,  nxl+1,nyl,  nzl  )]
                                            - udata->xflux[BUFIDX(2,i,  j,  k,  nxl+1,nyl,  nzl  )])/(udata->dx)
                                          + ( udata->yflux[BUFIDX(2,i,  j+1,k,  nxl,  nyl+1,nzl  )]
                                            - udata->yflux[BUFIDX(2,i,  j,  k,  nxl,  nyl+1,nzl  )])/(udata->dy)
                                          + ( udata->zflux[BUFIDX(2,i,  j,  k+1,nxl,  nyl,  nzl+1)]
                                            - udata->zflux[BUFIDX(2,i,  j,  k,  nxl,  nyl,  nzl+1)])/(udata->dz) );
        mzdot[ IDX(i,j,k,nxl,nyl,nzl)] -= ( ( udata->xflux[BUFIDX(3,i+1,j,  k,  nxl+1,nyl,  nzl  )]
                                            - udata->xflux[BUFIDX(3,i,  j,  k,  nxl+1,nyl,  nzl  )])/(udata->dx)
                                          + ( udata->yflux[BUFIDX(3,i,  j+1,k,  nxl,  nyl+1,nzl  )]
                                            - udata->yflux[BUFIDX(3,i,  j,  k,  nxl,  nyl+1,nzl  )])/(udata->dy)
                                          + ( udata->zflux[BUFIDX(3,i,  j,  k+1,nxl,  nyl,  nzl+1)]
                                            - udata->zflux[BUFIDX(3,i,  j,  k,  nxl,  nyl,  nzl+1)])/(udata->dz) );
        etdot[ IDX(i,j,k,nxl,nyl,nzl)] -= ( ( udata->xflux[BUFIDX(4,i+1,j,  k,  nxl+1,nyl,  nzl  )]
                                            - udata->xflux[BUFIDX(4,i,  j,  k,  nxl+1,nyl,  nzl  )])/(udata->dx)
                                          + ( udata->yflux[BUFIDX(4,i,  j+1,k,  nxl,  nyl+1,nzl  )]
                                            - udata->yflux[BUFIDX(4,i,  j,  k,  nxl,  nyl+1,nzl  )])/(udata->dy)
                                          + ( udata->zflux[BUFIDX(4,i,  j,  k+1,nxl,  nyl,  nzl+1)]
                                            - udata->zflux[BUFIDX(4,i,  j,  k,  nxl,  nyl,  nzl+1)])/(udata->dz) );
      }


  // return with success
  return 0;
}


// stability routine to compute maximum stable step size
static int stability(N_Vector w, realtype t, realtype* dt_stab, void* user_data)
{
  // access problem data
  UserData *udata = (UserData *) user_data;

  // access data arrays
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (stability)", 0)) return -1;

  // iterate over subdomain, computing the maximum local wave speed
  realtype u, p, csnd;
  realtype alpha = ZERO;
  for (long int i=0; i<(udata->nxl)*(udata->nyl)*(udata->nzl); i++) {
    u = max( max( abs(mx[i]/rho[i]), abs(mx[i]/rho[i]) ), abs(mx[i]/rho[i]) );
    p = udata->eos(rho[i], mx[i], my[i], mz[i], et[i]);
    csnd = SUNRsqrt((udata->gamma) * p / rho[i]);
    alpha = max(alpha, abs(u+csnd));
  }

  // determine maximum wave speed over entire domain
  int retval = MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_SUNREALTYPE, MPI_MAX, udata->comm);
  if (check_flag(&retval, "MPI_Alleduce (stability)", 3)) MPI_Abort(udata->comm, 1);

  // compute maximum stable step size
  *dt_stab = (udata->cfl) * min(min(udata->dx, udata->dy), udata->dz) / alpha;

  //printf("stab: alpha = %g, dt_stab = %g\n", alpha, *dt_stab);

  // return with success
  return(0);
}



//-----------------------------------------------------------
// Utility routines
//-----------------------------------------------------------


// Check function return value...
//  opt == 0 means SUNDIALS function allocates memory so check if
//           returned NULL pointer
//  opt == 1 means SUNDIALS function returns a flag so check if
//           flag >= 0
//  opt == 2 means function allocates memory so check if returned
//           NULL pointer
//  opt == 3 means MPI function returns a flag, so check if
//           flag != MPI_SUCCESS
//  opt == 4 corresponds to a check for an illegal state, so check if
//           flag >= 0
int check_flag(const void *flagvalue, const string funcname, const int opt)
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



// given a 6-point stencil of solution values,
//   w(x_{j-2}) w(x_{j-1}), w(x_j), w(x_{j+1}), w(x_{j+2}), w(x_{j+3})
// and the flux direction idir, compute the face-centered flux (dw)
// at the center of the stencil, x_{j+1/2}.
//
// The input "idir" handles the directionality for the 1D calculation
//    idir = 0  implies x-directional flux
//    idir = 1  implies y-directional flux
//    idir = 2  implies z-directional flux
//
// This precisely follows the recipe laid out in:
// Chi-Wang Shu (2003) "High-order Finite Difference and Finite Volume WENO
// Schemes and Discontinuous Galerkin Methods for CFD," International Journal of
// Computational Fluid Dynamics, 17:2, 107-118, DOI: 10.1080/1061856031000104851
void face_flux(realtype (&w1d)[6][NVAR], const int& idir,
               realtype* f_face, const UserData& udata)
{
  // local data
  int i, j;
  realtype rhosqrL, rhosqrR, rhosqrbar, u, v, w, H, qsq, csnd, cinv, cisq, gamm, alpha,
    beta1, beta2, beta3, w1, w2, w3, f1, f2, f3;
  realtype RV[NVAR][NVAR], LV[NVAR][NVAR], p[6], flux[6][NVAR], fproj[5][NVAR],
    fs[5][NVAR], fp[NVAR], fm[NVAR];
  const realtype bc = RCONST(1.083333333333333333333333333333333333333);    // 13/12
  const realtype dm[3] = {RCONST(0.1), RCONST(0.6), RCONST(0.3)};
  const realtype cm[3][3] = {{RCONST(1.833333333333333333333333333333333333333),    // 11/6
                              -RCONST(1.166666666666666666666666666666666666667),   // -7/6
                              RCONST(0.3333333333333333333333333333333333333333)},  // 1/3
                             {RCONST(0.3333333333333333333333333333333333333333),   // 1/3
                              RCONST(0.8333333333333333333333333333333333333333),   // 5/6
                              -RCONST(0.1666666666666666666666666666666666666667)}, // -1/6
                             {-RCONST(0.1666666666666666666666666666666666666667),  // -1/6
                              RCONST(0.8333333333333333333333333333333333333333),   // 5/6
                              RCONST(0.3333333333333333333333333333333333333333)}}; // 1/3
  const realtype dp[3] = {RCONST(0.3), RCONST(0.6), RCONST(0.1)};
  const realtype cp[3][3] = {{RCONST(0.3333333333333333333333333333333333333333),   // 1/3
                              RCONST(0.8333333333333333333333333333333333333333),   // 5/6
                              -RCONST(0.1666666666666666666666666666666666666667)}, // -1/6
                             {-RCONST(0.1666666666666666666666666666666666666667),  // -1/6
                              RCONST(0.8333333333333333333333333333333333333333),   // 5/6
                              RCONST(0.3333333333333333333333333333333333333333)},  // 1/3
                             {RCONST(0.3333333333333333333333333333333333333333),   // 1/3
                              -RCONST(1.166666666666666666666666666666666666667),   // -7/6
                              RCONST(1.833333333333333333333333333333333333333)}};  // 11/6
  const realtype epsilon = 1e-6;

  // convert state to direction-independent version
  if (idir > 0)
    for (i=0; i<6; i++) swap(w1d[i][1], w1d[i][1+idir]);

  // compute pressures over stencil
  for (i=0; i<6; i++)  p[i] = udata.eos(w1d[i][0], w1d[i][1], w1d[i][2], w1d[i][3], w1d[i][4]);

  // compute Roe-average state at face:
  //   wbar = [sqrt(rho), sqrt(rho)*vx, sqrt(rho)*vy, sqrt(rho)*vz, (e+p)/sqrt(rho)]
  //          [sqrt(rho), mx/sqrt(rho), my/sqrt(rho), mz/sqrt(rho), (e+p)/sqrt(rho)]
  //   u = wbar_2 / wbar_1
  //   v = wbar_3 / wbar_1
  //   w = wbar_4 / wbar_1
  //   H = wbar_5 / wbar_1
  rhosqrL = SUNRsqrt(w1d[2][0]);
  rhosqrR = SUNRsqrt(w1d[3][0]);
  rhosqrbar = RCONST(0.5)*(rhosqrL + rhosqrR);
  u = RCONST(0.5)*(w1d[2][1]/rhosqrL + w1d[3][1]/rhosqrR)/rhosqrbar;
  v = RCONST(0.5)*(w1d[2][2]/rhosqrL + w1d[3][2]/rhosqrR)/rhosqrbar;
  w = RCONST(0.5)*(w1d[2][3]/rhosqrL + w1d[3][3]/rhosqrR)/rhosqrbar;
  H = RCONST(0.5)*((p[2]+w1d[2][4])/rhosqrL + (p[3]+w1d[3][4])/rhosqrR)/rhosqrbar;

  // compute eigenvectors at face:
  qsq = u*u + v*v + w*w;
  gamm = udata.gamma-ONE;
  csnd = gamm*(H - RCONST(0.5)*qsq);
  cinv = ONE/csnd;
  cisq = cinv*cinv;
  for (i=0; i<NVAR; i++)
    for (j=0; j<NVAR; j++) {
      RV[i][j] = ZERO;
      LV[i][j] = ZERO;
    }

  RV[0][0] = ONE;
  RV[0][3] = ONE;
  RV[0][4] = ONE;

  RV[1][0] = u-csnd;
  RV[1][3] = u;
  RV[1][4] = u+csnd;

  RV[2][0] = v;
  RV[2][1] = ONE;
  RV[2][3] = v;
  RV[2][4] = v;

  RV[3][0] = w;
  RV[3][2] = ONE;
  RV[3][3] = w;
  RV[3][4] = w;

  RV[4][0] = H-u*csnd;
  RV[4][1] = v;
  RV[4][2] = w;
  RV[4][3] = HALF*qsq;
  RV[4][4] = H+u*csnd;

  LV[0][0] = HALF*cinv*(u + HALF*gamm*qsq);
  LV[0][1] = -HALF*cinv*(gamm*u + ONE);
  LV[0][2] = -HALF*v*gamm*cinv;
  LV[0][3] = -HALF*w*gamm*cinv;
  LV[0][4] = HALF*gamm*cinv;

  LV[1][0] = -v;
  LV[1][2] = ONE;

  LV[2][0] = -w;
  LV[2][3] = ONE;

  LV[3][0] = -gamm*cinv*(qsq - H);
  LV[3][1] = u*gamm*cinv;
  LV[3][2] = v*gamm*cinv;
  LV[3][3] = w*gamm*cinv;
  LV[3][4] = -gamm*cinv;

  LV[4][0] = -HALF*cinv*(u - HALF*gamm*qsq);
  LV[4][1] = -HALF*cinv*(gamm*u - ONE);
  LV[4][2] = -HALF*v*gamm*cinv;
  LV[4][3] = -HALF*w*gamm*cinv;
  LV[4][4] = HALF*gamm*cinv;

  // compute fluxes and max wave speed over stencil
  alpha = ZERO;
  for (i=0; i<6; i++) {
    u = w1d[i][1]/w1d[i][0];
    flux[i][0] = w1d[i][1];                       // mx
    flux[i][1] = u*w1d[i][1] + p[i];              // rho*vx*vx + p = mx*u + p
    flux[i][2] = u*w1d[i][2];                     // rho*vx*vy = my*u
    flux[i][3] = u*w1d[i][3];                     // rho*vx*vz = mz*u
    flux[i][4] = u*(w1d[i][4] + p[i]);            // vx*(et + p) = u*(et + p)
    csnd = SUNRsqrt(udata.gamma*p[i]/w1d[i][0]);  // c = sqrt(gamma*p/rho)
    alpha = max(alpha, abs(u)+csnd);
  }

  // fp(x_{i+1/2}):

  //   compute right-shifted Lax-Friedrichs flux over left portion of patch
  for (j=0; j<5; j++)
    for (i=0; i<NVAR; i++)
      fs[j][i] = HALF*(flux[j][i] + alpha*w1d[j][i]);

  // compute projected flux
  for (j=0; j<5; j++)
    for (i=0; i<NVAR; i++)
      fproj[j][i] = LV[i][0]*fs[j][0] + LV[i][1]*fs[j][1] + LV[i][2]*fs[j][2]
                  + LV[i][3]*fs[j][3] + LV[i][4]*fs[j][4];

  //   compute WENO signed fluxes
  for (i=0; i<NVAR; i++) {
    // smoothness indicators
    beta1 = bc*pow(fproj[2][i] - RCONST(2.0)*fproj[3][i] + fproj[4][i],2)
          + FOURTH*pow(RCONST(3.0)*fproj[2][i] - RCONST(4.0)*fproj[3][i] + fproj[4][i],2);
    beta2 = bc*pow(fproj[1][i] - RCONST(2.0)*fproj[2][i] + fproj[3][i],2)
          + FOURTH*pow(fproj[1][i] - fproj[3][i],2);
    beta3 = bc*pow(fproj[0][i] - RCONST(2.0)*fproj[1][i] + fproj[2][i],2)
          + FOURTH*pow(fproj[0][i] - RCONST(4.0)*fproj[1][i] + RCONST(3.0)*fproj[2][i],2);
    // nonlinear weights
    w1 = dp[0] / (epsilon + beta1) / (epsilon + beta1);
    w2 = dp[1] / (epsilon + beta2) / (epsilon + beta2);
    w3 = dp[2] / (epsilon + beta3) / (epsilon + beta3);
    // flux stencils
    f1 = cp[0][0]*fproj[2][i] + cp[0][1]*fproj[3][i] + cp[0][2]*fproj[4][i];
    f2 = cp[1][0]*fproj[1][i] + cp[1][1]*fproj[2][i] + cp[1][2]*fproj[3][i];
    f3 = cp[2][0]*fproj[0][i] + cp[2][1]*fproj[1][i] + cp[2][2]*fproj[2][i];
    // resulting signed flux
    fp[i] = (f1*w1 + f2*w2 + f3*w3)/(w1 + w2 + w3);
  }

  // fm(x_{i+1/2}):
  
  //   compute left-shifted Lax-Friedrichs flux over right portion of patch
  for (j=0; j<5; j++)
    for (i=0; i<NVAR; i++)
      fs[j][i] = HALF*(flux[j+1][i] - alpha*w1d[j+1][i]);

  // compute projected flux
  for (j=0; j<5; j++)
    for (i=0; i<NVAR; i++)
      fproj[j][i] = LV[i][0]*fs[j][0] + LV[i][1]*fs[j][1] + LV[i][2]*fs[j][2]
                  + LV[i][3]*fs[j][3] + LV[i][4]*fs[j][4];

  //   compute WENO signed fluxes
  for (i=0; i<NVAR; i++) {
    // smoothness indicators
    beta1 = bc*pow(fproj[2][i] - RCONST(2.0)*fproj[3][i] + fproj[4][i],2)
          + FOURTH*pow(RCONST(3.0)*fproj[2][i] - RCONST(4.0)*fproj[3][i] + fproj[4][i],2);
    beta2 = bc*pow(fproj[1][i] - RCONST(2.0)*fproj[2][i] + fproj[3][i],2)
          + FOURTH*pow(fproj[1][i] - fproj[3][i],2);
    beta3 = bc*pow(fproj[0][i] - RCONST(2.0)*fproj[1][i] + fproj[2][i],2)
          + FOURTH*pow(fproj[0][i] - RCONST(4.0)*fproj[1][i] + RCONST(3.0)*fproj[2][i],2);
    // nonlinear weights
    w1 = dm[0] / (epsilon + beta1) / (epsilon + beta1);
    w2 = dm[1] / (epsilon + beta2) / (epsilon + beta2);
    w3 = dm[2] / (epsilon + beta3) / (epsilon + beta3);
    // flux stencils
    f1 = cm[0][0]*fproj[2][i] + cm[0][1]*fproj[3][i] + cm[0][2]*fproj[4][i];
    f2 = cm[1][0]*fproj[1][i] + cm[1][1]*fproj[2][i] + cm[1][2]*fproj[3][i];
    f3 = cm[2][0]*fproj[0][i] + cm[2][1]*fproj[1][i] + cm[2][2]*fproj[2][i];
    // resulting signed flux
    fm[i] = (f1*w1 + f2*w2 + f3*w3)/(w1 + w2 + w3);
  }

  // combine signed fluxes into output, converting back to conserved variables
  for (i=0; i<NVAR; i++)
    f_face[i] = RV[i][0]*(fm[0] + fp[0]) + RV[i][1]*(fm[1] + fp[1])
              + RV[i][2]*(fm[2] + fp[2]) + RV[i][3]*(fm[3] + fp[3])
              + RV[i][4]*(fm[4] + fp[4]);

  // convert fluxes to direction-independent version
  if (idir > 0) 
    swap(f_face[1], f_face[1+idir]);

}

//---- end of file ----
