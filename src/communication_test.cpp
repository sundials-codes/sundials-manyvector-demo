/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Main routine to test communication infrastructure.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <iomanip>

#include "fenv.h"

// Main Program
int main(int argc, char* argv[]) {

  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

  // general problem parameters
  long int N, Ntot, Nsubvecs, i, j, k, v, idx;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int myid;                      // MPI process ID
  N_Vector w = NULL;             // empty vectors for storing overall solution
  N_Vector *wsubvecs;

  // initialize MPI
  retval = MPI_Init(&argc, &argv);
  if (check_flag(&retval, "MPI_Init (main)", 3)) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (check_flag(&retval, "MPI_Comm_rank (main)", 3)) MPI_Abort(MPI_COMM_WORLD, 1);

  // ensure that realtype is at least double precision
  if (sizeof(realtype) < sizeof(double)) {
    if (myid == 0)
      cerr << "Communication_test error: incompatible precision.\n"
           << "Test must be run with realtype at least double precision.\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  
  // read problem and solver parameters from input file / command line
  UserData udata;
  ARKodeParameters opts;
  retval = load_inputs(myid, argc, argv, udata, opts);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (retval > 0) MPI_Abort(MPI_COMM_WORLD, 0);

  // overwrite specified boundary conditions to enable periodicity in all directions
  udata.xlbc = udata.xrbc = udata.ylbc = udata.yrbc = udata.zlbc = udata.zrbc = 0;
  
  // set up udata structure
  retval = udata.SetupDecomp();
  if (check_flag(&retval, "SetupDecomp (main)", 1)) MPI_Abort(udata.comm, 1);

  // ensure that specified grid has global extents no larger than 1000
  // (so that global indices fit in 3 digits)
  if ((udata.nx > 1000) || (udata.ny > 1000) || (udata.nz > 1000)) {
    if (udata.myid == 0)
      cerr << "Communication_test error: incompatible mesh.\n"
           << "Global extents in each direction must fit within 3 digits.\n"
           << "(" << udata.nx << " x " << udata.ny << " x " << udata.nz << " used here)\n";
    MPI_Abort(udata.comm, 1);
  }
  
  // ensure that test uses at most 100 MPI tasks
  // (so that MPI indices fit in 2 digits)
  if (udata.nprocs > 100) {
    if (udata.myid == 0)
      cerr << "Communication_test error: test must be run with at most 100 tasks.\n"
           << "(" << udata.nprocs << " used here)\n.";
    MPI_Abort(udata.comm, 1);
  }
  
  // ensure that test uses at most 10 total unknowns per spatial location
  // (so that species indices fit in 1 digit)
  if (NVAR > 9) {
    if (udata.myid == 0)
      cerr << "Communication_test error: test must be run with at most 10\n"
           << "unknowns per spatial location. (" << NVAR << " used here)\n.";
    MPI_Abort(udata.comm, 1);
  }
  
  // Initial problem output
  bool outproc = (udata.myid == 0);
  if (outproc) {
    cout << "\nCommunication subsystem testing program:\n";
    cout << "   nprocs: " << udata.nprocs << " (" << udata.npx << " x "
         << udata.npy << " x " << udata.npz << ")\n";
    cout << "   spatial grid: " << udata.nx << " x " << udata.ny << " x "
         << udata.nz << "\n";
    cout << "   tracers/chemical species: " << udata.nchem << "\n";
    cout << "   bdry cond (0=per, 1=Neu, 2=Dir): ["
         << udata.xlbc << ", " << udata.xrbc << "] x ["
         << udata.ylbc << ", " << udata.yrbc << "] x ["
         << udata.zlbc << ", " << udata.zrbc << "]\n";
  }
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  printf("      proc %4i: %li x %li x %li\n", udata.myid, udata.nxl, udata.nyl, udata.nzl);
  retval = MPI_Barrier(udata.comm);
  if (check_flag(&retval, "MPI_Barrier (main)", 3)) MPI_Abort(udata.comm, 1);
  
  // Initialize N_Vector data structures
  N = (udata.nxl)*(udata.nyl)*(udata.nzl);
  Ntot = (udata.nx)*(udata.ny)*(udata.nz);
  Nsubvecs = 5 + udata.nchem*N;
  wsubvecs = new N_Vector[Nsubvecs];
  for (i=0; i<5; i++) {
    wsubvecs[i] = NULL;
    wsubvecs[i] = N_VNew_Parallel(udata.comm, N, Ntot);
    if (check_flag((void *) wsubvecs[i], "N_VNew_Parallel (main)", 0)) MPI_Abort(udata.comm, 1);
  }
  for (i=5; i<Nsubvecs; i++) {
    wsubvecs[i] = NULL;
    wsubvecs[i] = N_VNew_Serial(udata.nchem);
    if (check_flag((void *) wsubvecs[i], "N_VNew_Serial (main)", 0)) MPI_Abort(udata.comm, 1);
  }
  w = N_VNew_MPIManyVector(Nsubvecs, wsubvecs);  // combined solution vector
  if (check_flag((void *) w, "N_VNew_MPIManyVector (main)", 0)) MPI_Abort(udata.comm, 1);

  // set data into subvectors:
  //   digits 1-2:   MPI task ID
  //   digit  3:     species index
  //   digits 4-6:   global x index
  //   digits 7-9:   global y index
  //   digits 10-12: global z index
  realtype myid_value = RCONST(0.01)*udata.myid;
  realtype *wdata;
  realtype species_value, xloc_value, yloc_value, zloc_value, true_value;
  //   first fill the fluid vectors
  for (v=0; v<5; v++) {
    wdata = N_VGetArrayPointer(wsubvecs[v]);
    if (check_flag((void *) wdata, "N_VGetArrayPointer (main)", 0)) return -1;
    species_value = RCONST(0.001)*v;
    for (k=0; k<udata.nzl; k++) 
      for (j=0; j<udata.nyl; j++) 
        for (i=0; i<udata.nxl; i++) {
          xloc_value = RCONST(0.000001)*(i+udata.is);
          yloc_value = RCONST(0.000000001)*(j+udata.js);
          zloc_value = RCONST(0.000000000001)*(k+udata.ks);
          wdata[IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl)] = myid_value + species_value
            + xloc_value + yloc_value + zloc_value;
        }
  }
  //   then fill the tracer vectors
  if (udata.nchem > 0) {
    for (k=0; k<udata.nzl; k++) 
      for (j=0; j<udata.nyl; j++) 
        for (i=0; i<udata.nxl; i++) {
          xloc_value = RCONST(0.000001)*(i+udata.is);
          yloc_value = RCONST(0.000000001)*(j+udata.js);
          zloc_value = RCONST(0.000000000001)*(k+udata.ks);
          idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
          wdata = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+idx);
          if (check_flag((void *) wdata, "N_VGetSubvectorArrayPointer_MPIManyVector (main)", 0)) return -1;
          for (v=0; v<udata.nchem; v++) {
            species_value = RCONST(0.001)*(5+v);
            wdata[v] = myid_value + species_value + xloc_value + yloc_value + zloc_value;
          }
        }
  }
  
  // perform communication
  retval = udata.ExchangeStart(w);
  if (check_flag(&retval, "ExchangeStart (main)", 1)) return -1;
  retval = udata.ExchangeEnd();
  if (check_flag(&retval, "ExchangeEnd (main)", 1)) return -1;


  // check receive buffers for accuracy
  long int loc_errs = 0;
  realtype test_tol = 1e-12;
  realtype recv_value;
  //    Wrecv
  myid_value = RCONST(0.01)*udata.ipW;
  long int ieW = (udata.is==0) ? udata.nx-1 : udata.is-1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++) 
      for (i=0; i<3; i++)
        for (v=0; v<NVAR; v++) {
          species_value = RCONST(0.001)*v;
          xloc_value = RCONST(0.000001)*(i+ieW-2);
          yloc_value = RCONST(0.000000001)*(j+udata.js);
          zloc_value = RCONST(0.000000000001)*(k+udata.ks);
          true_value = myid_value + species_value
                     + xloc_value + yloc_value + zloc_value;
          recv_value = udata.Wrecv[BUFIDX(v,i,j,k,3,udata.nyl,udata.nzl)];
          if (abs(recv_value-true_value) > test_tol) {
            cout << "Wrecv error: myid = " << udata.myid << ", (v,i,j,k) = ("
                 << v << ", " << i << ", " << j << ", " << k << "), Wrecv = "
                 << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
            loc_errs++;
          }
        }

  //    Erecv
  myid_value = RCONST(0.01)*udata.ipE;
  long int isE = (udata.ie==udata.nx-1) ? 0 : udata.ie+1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++) 
      for (i=0; i<3; i++)
        for (v=0; v<NVAR; v++) {
          species_value = RCONST(0.001)*v;
          xloc_value = RCONST(0.000001)*(i+isE);
          yloc_value = RCONST(0.000000001)*(j+udata.js);
          zloc_value = RCONST(0.000000000001)*(k+udata.ks);
          true_value = myid_value + species_value
                     + xloc_value + yloc_value + zloc_value;
          recv_value = udata.Erecv[BUFIDX(v,i,j,k,3,udata.nyl,udata.nzl)];
          if (abs(recv_value-true_value) > test_tol) {
            cout << "Erecv error: myid = " << udata.myid << ", (v,i,j,k) = ("
                 << v << ", " << i << ", " << j << ", " << k << "), Erecv = "
                 << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
            loc_errs++;
          }
        }

  //    Srecv
  myid_value = RCONST(0.01)*udata.ipS;
  long int jeS = (udata.js==0) ? udata.ny-1 : udata.js-1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<3; j++) 
      for (i=0; i<udata.nxl; i++)
        for (v=0; v<NVAR; v++) {
          species_value = RCONST(0.001)*v;
          xloc_value = RCONST(0.000001)*(i+udata.is);
          yloc_value = RCONST(0.000000001)*(j+jeS-2);
          zloc_value = RCONST(0.000000000001)*(k+udata.ks);
          true_value = myid_value + species_value
                     + xloc_value + yloc_value + zloc_value;
          recv_value = udata.Srecv[BUFIDX(v,j,i,k,3,udata.nxl,udata.nzl)];
          if (abs(recv_value-true_value) > test_tol) {
            cout << "Srecv error: myid = " << udata.myid << ", (v,i,j,k) = ("
                 << v << ", " << i << ", " << j << ", " << k << "), Srecv = "
                 << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
            loc_errs++;
          }
        }

  //    Nrecv
  myid_value = RCONST(0.01)*udata.ipN;
  long int jsN = (udata.je==udata.ny-1) ? 0 : udata.je+1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<3; j++) 
      for (i=0; i<udata.nxl; i++)
        for (v=0; v<NVAR; v++) {
          species_value = RCONST(0.001)*v;
          xloc_value = RCONST(0.000001)*(i+udata.is);
          yloc_value = RCONST(0.000000001)*(j+jsN);
          zloc_value = RCONST(0.000000000001)*(k+udata.ks);
          true_value = myid_value + species_value
                     + xloc_value + yloc_value + zloc_value;
          recv_value = udata.Nrecv[BUFIDX(v,j,i,k,3,udata.nxl,udata.nzl)];
          if (abs(recv_value-true_value) > test_tol) {
            cout << "Nrecv error: myid = " << udata.myid << ", (v,i,j,k) = ("
                 << v << ", " << i << ", " << j << ", " << k << "), Nrecv = "
                 << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
            loc_errs++;
          }
        }

  //    Brecv
  myid_value = RCONST(0.01)*udata.ipB;
  long int keB = (udata.ks==0) ? udata.nz-1 : udata.ks-1;
  for (k=0; k<3; k++)
    for (j=0; j<udata.nyl; j++) 
      for (i=0; i<udata.nxl; i++)
        for (v=0; v<NVAR; v++) {
          species_value = RCONST(0.001)*v;
          xloc_value = RCONST(0.000001)*(i+udata.is);
          yloc_value = RCONST(0.000000001)*(j+udata.js);
          zloc_value = RCONST(0.000000000001)*(k+keB-2);
          true_value = myid_value + species_value
                     + xloc_value + yloc_value + zloc_value;
          recv_value = udata.Brecv[BUFIDX(v,k,i,j,3,udata.nxl,udata.nyl)];
          if (abs(recv_value-true_value) > test_tol) {
            cout << "Brecv error: myid = " << udata.myid << ", (v,i,j,k) = ("
                 << v << ", " << i << ", " << j << ", " << k << "), Brecv = "
                 << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
            loc_errs++;
          }
        }

  //    Frecv
  myid_value = RCONST(0.01)*udata.ipF;
  long int ksF = (udata.ke==udata.nz-1) ? 0 : udata.ke+1;
  for (k=0; k<3; k++)
    for (j=0; j<udata.nyl; j++) 
      for (i=0; i<udata.nxl; i++)
        for (v=0; v<NVAR; v++) {
          species_value = RCONST(0.001)*v;
          xloc_value = RCONST(0.000001)*(i+udata.is);
          yloc_value = RCONST(0.000000001)*(j+udata.js);
          zloc_value = RCONST(0.000000000001)*(k+ksF);
          true_value = myid_value + species_value
                     + xloc_value + yloc_value + zloc_value;
          recv_value = udata.Frecv[BUFIDX(v,k,i,j,3,udata.nxl,udata.nyl)];
          if (abs(recv_value-true_value) > test_tol) {
            cout << "Frecv error: myid = " << udata.myid << ", (v,i,j,k) = ("
                 << v << ", " << i << ", " << j << ", " << k << "), Frecv = "
                 << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
            loc_errs++;
          }
        }

  // report on total errors encountered
  long int tot_errs = 0;
  retval = MPI_Reduce(&loc_errs, &tot_errs, 1, MPI_LONG, MPI_SUM, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (main)", 3)) return(1);
  if (udata.myid == 0) 
    cout << "Communication_test result: " << tot_errs << " total errors\n";
  
  // Clean up and return with successful completion
  N_VDestroy(w);               // Free solution vectors
  for (i=0; i<Nsubvecs; i++)
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  MPI_Finalize();              // Finalize MPI
  return 0;
}

//---- end of file ----
