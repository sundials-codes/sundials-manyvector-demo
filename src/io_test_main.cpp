/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Main routine to test HDF5 output/input infrastructure.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <iomanip>

#include "fenv.h"

// Main Program
int main(int argc, char* argv[]) {

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem parameters
  long int N, Ntot, i, j, k, l, v, idx;
  int Nsubvecs;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use
  N_Vector w, wtest, werr;       // empty vectors for storing overall solution
  N_Vector *wsubvecs;
  realtype tout;
  w = wtest = werr = NULL;

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // ensure that realtype is at least double precision
  if (sizeof(realtype) < sizeof(double)) {
    if (myid == 0)
      cerr << "io_test error: incompatible precision.\n"
           << "Test must be run with realtype at least double precision.\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // read problem and solver parameters from input file / command line
  EulerData udata;
  ARKodeParameters opts;
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (retval > 0) MPI_Abort(MPI_COMM_WORLD, 0);

  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);

  // Output problem setup information
  bool outproc = (udata.myid == 0);
  if (outproc) {
    cout << "\nFile I/O subsystem testing program:\n";
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
  }
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  printf("      proc %4i: %li x %li x %li\n", udata.myid, udata.nxl, udata.nyl, udata.nzl);
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);

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
#ifdef USERAJA
    wsubvecs[5] = N_VNewManaged_Raja(N*udata.nchem);
    if (check_flag((void *) wsubvecs[5], "N_VNewManaged_Raja (main)", 0)) MPI_Abort(udata.comm, 1);
#else
    wsubvecs[5] = N_VNew_Serial(N*udata.nchem);
    if (check_flag((void *) wsubvecs[5], "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
#endif
  }
  w = N_VNew_MPIManyVector(Nsubvecs, wsubvecs);  // combined solution vector
  if (check_flag((void *) w, "N_VNew_MPIManyVector (main)", 0)) MPI_Abort(udata.comm, 1);
  wtest = N_VClone(w);
  if (check_flag((void *) wtest, "N_VClone (main)", 0)) MPI_Abort(udata.comm, 1);
  werr = N_VClone(w);
  if (check_flag((void *) werr, "N_VClone (main)", 0)) MPI_Abort(udata.comm, 1);

  // set data into subvectors:
  //   digit  1:    species index
  //   digits 2-4:  global x index
  //   digits 5-7:  global y index
  //   digits 8-10: global z index
  if (udata.myid == 0)
    cout << "\nAll data uses the following numbering conventions:\n"
         << "   digit  1:     species index (s)\n"
         << "   digits 2-4:   global x index (X)\n"
         << "   digits 5-7:   global y index (y)\n"
         << "   digits 8-10:  global z index (Z)\n"
         << "i.e., data value digit convention:\n"
         << "   0.sXXXyyyZZZ\n";
  realtype *wdata;
  realtype species_value, xloc_value, yloc_value, zloc_value, true_value;
  //   first fill the fluid vectors
  for (v=0; v<5; v++) {
    wdata = N_VGetArrayPointer(wsubvecs[v]);
    if (check_flag((void *) wdata, "N_VGetArrayPointer (main)", 0)) return -1;
    for (k=0; k<udata.nzl; k++)
      for (j=0; j<udata.nyl; j++)
        for (i=0; i<udata.nxl; i++) {
          species_value = RCONST(0.001)*v;
          xloc_value = RCONST(0.0001)*(i+udata.is);
          yloc_value = RCONST(0.0000001)*(j+udata.js);
          zloc_value = RCONST(0.0000000001)*(k+udata.ks);
          idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
          wdata[idx] = species_value + xloc_value + yloc_value + zloc_value;
        }
  }
  //   then fill the tracer vectors
  if (udata.nchem > 0) {
    wdata = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
    if (check_flag((void *) wdata, "N_VGetSubvectorArrayPointer_MPIManyVector (main)", 0)) return -1;
    for (k=0; k<udata.nzl; k++)
      for (j=0; j<udata.nyl; j++)
        for (i=0; i<udata.nxl; i++) {
          for (v=0; v<udata.nchem; v++) {
            species_value = RCONST(0.001)*(5+v);
            xloc_value = RCONST(0.0001)*(i+udata.is);
            yloc_value = RCONST(0.0000001)*(j+udata.js);
            zloc_value = RCONST(0.0000000001)*(k+udata.ks);
            idx = BUFIDX(v,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
            wdata[idx] = species_value + xloc_value + yloc_value + zloc_value;
          }
        }
  }

  // if running test from scratch, output w to disk
  if (restart < 0) {
    restart = 5;
    if (udata.myid == 0)  cout << "\nSaving current state to restart file " << restart << endl;
    tout = RCONST(8.5);
    retval = output_solution(tout, w, opts.h0, restart, udata, opts);
    if (check_flag(&retval, "output_solution (main)", 1)) MPI_Abort(udata.comm, 1);
  }

  // read w from disk
  if (udata.myid == 0)  cout << "\nReading state from restart file " << restart << endl;
  retval = read_restart(restart, tout, wtest, udata);
  if (check_flag(&retval, "read_restart (main)", 1)) MPI_Abort(udata.comm, 1);

  // check tout and overall norm of solution difference
  if (udata.myid == 0)  cout << "\nChecking I/O results\n";
  long int loc_errs = 0;
  if ((udata.myid == 0) && (abs(tout - RCONST(8.5)) > 1e-14)) {
    cout << fixed << setprecision(14) << "  restart 'time' error: "
         << tout << " != " << RCONST(8.5) << endl;
    loc_errs++;
  }
  N_VLinearSum(-ONE, w, ONE, wtest, werr);
  realtype werr_max = N_VMaxNorm(werr);
  realtype test_tol = 1e-12;
  if (werr_max > test_tol) {

    cout << fixed << setprecision(14) << "  restart state max error: "
         << werr_max << " > " << test_tol << ", examining more closely:\n";

    // solution vector shows differences; examine more closely
    //   first the fluid vectors
    for (v=0; v<5; v++) {
      wdata = N_VGetArrayPointer(wsubvecs[v]);
      if (check_flag((void *) wdata, "N_VGetArrayPointer (main)", 0)) return -1;
      for (k=0; k<udata.nzl; k++)
        for (j=0; j<udata.nyl; j++)
          for (i=0; i<udata.nxl; i++) {
            species_value = RCONST(0.001)*v;
            xloc_value = RCONST(0.0001)*(i+udata.is);
            yloc_value = RCONST(0.0000001)*(j+udata.js);
            zloc_value = RCONST(0.0000000001)*(k+udata.ks);
            idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
            realtype true_value = species_value + xloc_value + yloc_value + zloc_value;
            realtype recv_value = wdata[idx];
            if (abs(recv_value-true_value) > test_tol) {
              cout << "    myid = " << udata.myid << ", (v,i,j,k) = ("
                   << v << ", " << i << ", " << j << ", " << k << "), w = "
                   << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
              loc_errs++;
            }
          }
    }
    //   then the tracer vectors
    if (udata.nchem > 0) {
      wdata = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
      if (check_flag((void *) wdata, "N_VGetSubvectorArrayPointer_MPIManyVector (main)", 0)) return -1;
      for (k=0; k<udata.nzl; k++)
        for (j=0; j<udata.nyl; j++)
          for (i=0; i<udata.nxl; i++) {
            for (v=0; v<udata.nchem; v++) {
              xloc_value = RCONST(0.0001)*(i+udata.is);
              yloc_value = RCONST(0.0000001)*(j+udata.js);
              zloc_value = RCONST(0.0000000001)*(k+udata.ks);
              species_value = RCONST(0.001)*(5+v);
              idx = BUFIDX(v,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
              realtype true_value = species_value + xloc_value + yloc_value + zloc_value;
              realtype recv_value = wdata[idx];
              if (abs(recv_value-true_value) > test_tol) {
                cout << "    myid = " << udata.myid << ", (v,i,j,k) = ("
                     << 5+v << ", " << i << ", " << j << ", " << k << "), w = "
                     << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
                loc_errs++;
              }
            }
          }
    }
  }

  // report on total errors encountered
  long int tot_errs = 0;
  retval = MPI_Reduce(&loc_errs, &tot_errs, 1, MPI_LONG, MPI_SUM, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (main)", 3)) return(1);
  if (udata.myid == 0)
    cout << "\nio_test result: " << tot_errs << " total errors\n";

  // Clean up and return with successful completion
  N_VDestroy(w);               // Free solution vectors
  N_VDestroy(wtest);
  N_VDestroy(werr);
  for (i=0; i<Nsubvecs; i++)
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  MPI_Finalize();              // Finalize MPI
  return 0;
}

//---- end of file ----
