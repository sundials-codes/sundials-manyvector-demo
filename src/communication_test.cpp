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

#ifdef DEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif

  // general problem parameters
  long int N, Ntot, i, j, k, l, v, idx;
  int Nsubvecs;

  // general problem variables
  int retval;                    // reusable error-checking flag
  int myid;                      // MPI process ID
  int restart;                   // restart file number to use (unused here)
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
  retval = load_inputs(myid, argc, argv, udata, opts, restart);
  if (check_flag(&retval, "load_inputs (main)", 1)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (retval > 0) MPI_Abort(MPI_COMM_WORLD, 0);

  // overwrite specified boundary conditions to enable periodicity in all directions
  udata.xlbc = udata.xrbc = udata.ylbc = udata.yrbc = udata.zlbc = udata.zrbc = BC_PERIODIC;

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
    cout << "   bdry cond (" << BC_PERIODIC << "=per, " << BC_NEUMANN << "=Neu, "
         << BC_DIRICHLET << "=Dir, " << BC_REFLECTING << "=refl): ["
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

  // set data into subvectors:
  //   digits 1-2:   MPI task ID
  //   digit  3:     species index
  //   digits 4-6:   global x index
  //   digits 7-9:   global y index
  //   digits 10-12: global z index
  if (udata.myid == 0)
    cout << "\nAll data uses the following numbering conventions:\n"
         << "   digits 1-2:   MPI task ID (T)\n"
         << "   digit  3:     species index (s)\n"
         << "   digits 4-6:   global x index (X)\n"
         << "   digits 7-9:   global y index (y)\n"
         << "   digits 10-12: global z index (Z)\n"
         << "i.e., data value digit convention:\n"
         << "   0.TTsXXXyyyZZZ\n\n";
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
          idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
          wdata[idx] = myid_value + species_value + xloc_value + yloc_value + zloc_value;
        }
  }
  //   then fill the tracer vectors
  if (udata.nchem > 0) {
    wdata = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
    if (check_flag((void *) wdata, "N_VGetSubvectorArrayPointer_MPIManyVector (main)", 0)) return -1;
    for (k=0; k<udata.nzl; k++)
      for (j=0; j<udata.nyl; j++)
        for (i=0; i<udata.nxl; i++)
          for (v=0; v<udata.nchem; v++) {
            species_value = RCONST(0.001)*(5+v);
            xloc_value = RCONST(0.000001)*(i+udata.is);
            yloc_value = RCONST(0.000000001)*(j+udata.js);
            zloc_value = RCONST(0.000000000001)*(k+udata.ks);
            idx = BUFIDX(v,i,j,k,udata.nchem,udata.nxl,udata.nyl,udata.nzl);
            wdata[idx] = myid_value + species_value + xloc_value + yloc_value + zloc_value;
          }
  }

  // perform communication
  retval = udata.ExchangeStart(w);
  if (check_flag(&retval, "ExchangeStart (main)", 1)) return -1;
  retval = udata.ExchangeEnd();
  if (check_flag(&retval, "ExchangeEnd (main)", 1)) return -1;


  // check receive buffers for accuracy
  if (udata.myid == 0)  cout << "\nChecking receive buffers for accuracy:\n";
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
          idx = BUFIDX(v,i,j,k,NVAR,3,udata.nyl,udata.nzl);
          recv_value = udata.Wrecv[idx];
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
          idx = BUFIDX(v,i,j,k,NVAR,3,udata.nyl,udata.nzl);
          recv_value = udata.Erecv[idx];
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
          idx = BUFIDX(v,j,i,k,NVAR,3,udata.nxl,udata.nzl);
          recv_value = udata.Srecv[idx];
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
          idx = BUFIDX(v,j,i,k,NVAR,3,udata.nxl,udata.nzl);
          recv_value = udata.Nrecv[idx];
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
          idx = BUFIDX(v,k,i,j,NVAR,3,udata.nxl,udata.nyl);
          recv_value = udata.Brecv[idx];
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
          idx = BUFIDX(v,k,i,j,NVAR,3,udata.nxl,udata.nyl);
          recv_value = udata.Frecv[idx];
          if (abs(recv_value-true_value) > test_tol) {
            cout << "Frecv error: myid = " << udata.myid << ", (v,i,j,k) = ("
                 << v << ", " << i << ", " << j << ", " << k << "), Frecv = "
                 << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
            loc_errs++;
          }
        }


  // now check "pack1D" routines for accuracy
  myid_value = RCONST(0.01)*udata.myid;
  if (udata.myid == 0)
    cout << "\nChecking pack1D routines for accuracy:\n";
  //     interior
  realtype w1d[6][NVAR];
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (main)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (main)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (main)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (main)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (main)", 0)) return -1;
  realtype *chem = NULL;
  if (udata.nchem > 0) {
    chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5);
    if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (main)", 0)) return -1;
  }
  for (k=3; k<udata.nzl-2; k++)
    for (j=3; j<udata.nyl-2; j++)
      for (i=3; i<udata.nxl-2; i++) {

        // x-directional stencil
        udata.pack1D_x(w1d, rho, mx, my, mz, et, chem, i, j, k);
        for (l=0; l<6; l++)
          for (v=0; v<NVAR; v++) {
            species_value = RCONST(0.001)*v;
            xloc_value = RCONST(0.000001)*(i+l-3+udata.is);
            yloc_value = RCONST(0.000000001)*(j+udata.js);
            zloc_value = RCONST(0.000000000001)*(k+udata.ks);
            true_value = myid_value + species_value
              + xloc_value + yloc_value + zloc_value;
            recv_value = w1d[l][v];
            if (abs(recv_value-true_value) > test_tol) {
              cout << "pack1D_x error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                   << v << ", " << i << ", " << j << ", " << k << ", " << l << "), w1d = "
                   << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
              loc_errs++;
            }
          }

        // y-directional stencil
        udata.pack1D_y(w1d, rho, mx, my, mz, et, chem, i, j, k);
        for (l=0; l<6; l++)
          for (v=0; v<NVAR; v++) {
            species_value = RCONST(0.001)*v;
            xloc_value = RCONST(0.000001)*(i+udata.is);
            yloc_value = RCONST(0.000000001)*(j+l-3+udata.js);
            zloc_value = RCONST(0.000000000001)*(k+udata.ks);
            true_value = myid_value + species_value
              + xloc_value + yloc_value + zloc_value;
            recv_value = w1d[l][v];
            if (abs(recv_value-true_value) > test_tol) {
              cout << "pack1D_y error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                   << v << ", " << i << ", " << j << ", " << k << ", " << l << "), w1d = "
                   << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
              loc_errs++;
            }
          }

        // z-directional stencil
        udata.pack1D_z(w1d, rho, mx, my, mz, et, chem, i, j, k);
        for (l=0; l<6; l++)
          for (v=0; v<NVAR; v++) {
            species_value = RCONST(0.001)*v;
            xloc_value = RCONST(0.000001)*(i+udata.is);
            yloc_value = RCONST(0.000000001)*(j+udata.js);
            zloc_value = RCONST(0.000000000001)*(k+l-3+udata.ks);
            true_value = myid_value + species_value
              + xloc_value + yloc_value + zloc_value;
            recv_value = w1d[l][v];
            if (abs(recv_value-true_value) > test_tol) {
              cout << "pack1D_z error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                   << v << ", " << i << ", " << j << ", " << k << ", " << l << "), w1d = "
                   << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
              loc_errs++;
            }
          }

      }

  //     boundary
  if (udata.myid == 0)
    cout << "\nChecking pack1D*bdry routines for accuracy:\n";
  long int istencil, jstencil, kstencil;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {

        // skip strict interior (already computed)
        if ( (k>2) && (k<udata.nzl-2) && (j>2) && (j<udata.nyl-2) && (i>2) && (i<udata.nxl-2) )
          continue;

        // x-directional stencil
        udata.pack1D_x_bdry(w1d, rho, mx, my, mz, et, chem, i, j, k);
        for (l=0; l<6; l++)
          for (v=0; v<NVAR; v++) {
            myid_value = RCONST(0.01)*udata.myid;
            if (i+l-3 < 0)            myid_value = RCONST(0.01)*udata.ipW;
            if (i+l-3 > udata.nxl-1)  myid_value = RCONST(0.01)*udata.ipE;
            species_value = RCONST(0.001)*v;
            istencil = i+l-3+udata.is;
            if (istencil < 0)           istencil += udata.nx;
            if (istencil > udata.nx-1)  istencil -= udata.nx;
            jstencil = j+udata.js;
            kstencil = k+udata.ks;
            xloc_value = RCONST(0.000001)*istencil;
            yloc_value = RCONST(0.000000001)*jstencil;
            zloc_value = RCONST(0.000000000001)*kstencil;
            true_value = myid_value + species_value
              + xloc_value + yloc_value + zloc_value;
            recv_value = w1d[l][v];
            if (abs(recv_value-true_value) > test_tol) {
              cout << "pack1D_x_bdry error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                   << v << ", " << i << ", " << j << ", " << k << ", " << l << "), w1d = "
                   << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
              loc_errs++;
            }
          }

        // x-directional stencil at upper boundary face
        if (i == udata.nxl-1) {
          udata.pack1D_x_bdry(w1d, rho, mx, my, mz, et, chem, udata.nxl, j, k);
          for (l=0; l<6; l++)
            for (v=0; v<NVAR; v++) {
              myid_value = RCONST(0.01)*udata.myid;
              if (udata.nxl+l-3 < 0)            myid_value = RCONST(0.01)*udata.ipW;
              if (udata.nxl+l-3 > udata.nxl-1)  myid_value = RCONST(0.01)*udata.ipE;
              species_value = RCONST(0.001)*v;
              istencil = udata.nxl+l-3+udata.is;
              if (istencil < 0)           istencil += udata.nx;
              if (istencil > udata.nx-1)  istencil -= udata.nx;
              jstencil = j+udata.js;
              kstencil = k+udata.ks;
              xloc_value = RCONST(0.000001)*istencil;
              yloc_value = RCONST(0.000000001)*jstencil;
              zloc_value = RCONST(0.000000000001)*kstencil;
              true_value = myid_value + species_value
                + xloc_value + yloc_value + zloc_value;
              recv_value = w1d[l][v];
              if (abs(recv_value-true_value) > test_tol) {
                cout << "pack1D_x_bdry error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                     << v << ", " << udata.nxl << ", " << j << ", " << k << ", " << l << "), w1d = "
                     << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
                loc_errs++;
              }
            }
        }

        // y-directional stencil
        udata.pack1D_y_bdry(w1d, rho, mx, my, mz, et, chem, i, j, k);
        for (l=0; l<6; l++)
          for (v=0; v<NVAR; v++) {
            myid_value = RCONST(0.01)*udata.myid;
            if (j+l-3 < 0)            myid_value = RCONST(0.01)*udata.ipS;
            if (j+l-3 > udata.nyl-1)  myid_value = RCONST(0.01)*udata.ipN;
            species_value = RCONST(0.001)*v;
            istencil = i+udata.is;
            jstencil = j+l-3+udata.js;
            if (jstencil < 0)           jstencil += udata.ny;
            if (jstencil > udata.ny-1)  jstencil -= udata.ny;
            kstencil = k+udata.ks;
            xloc_value = RCONST(0.000001)*istencil;
            yloc_value = RCONST(0.000000001)*jstencil;
            zloc_value = RCONST(0.000000000001)*kstencil;
            true_value = myid_value + species_value
              + xloc_value + yloc_value + zloc_value;
            recv_value = w1d[l][v];
            if (abs(recv_value-true_value) > test_tol) {
              cout << "pack1D_y_bdry error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                   << v << ", " << i << ", " << j << ", " << k << ", " << l << "), w1d = "
                   << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
              loc_errs++;
            }
          }

        // y-directional stencil at upper boundary face
        if (j == udata.nyl-1) {
          udata.pack1D_y_bdry(w1d, rho, mx, my, mz, et, chem, i, udata.nyl, k);
          for (l=0; l<6; l++)
            for (v=0; v<NVAR; v++) {
              myid_value = RCONST(0.01)*udata.myid;
              if (udata.nyl+l-3 < 0)            myid_value = RCONST(0.01)*udata.ipS;
              if (udata.nyl+l-3 > udata.nyl-1)  myid_value = RCONST(0.01)*udata.ipN;
              species_value = RCONST(0.001)*v;
              istencil = i+udata.is;
              jstencil = udata.nyl+l-3+udata.js;
              if (jstencil < 0)           jstencil += udata.ny;
              if (jstencil > udata.ny-1)  jstencil -= udata.ny;
              kstencil = k+udata.ks;
              xloc_value = RCONST(0.000001)*istencil;
              yloc_value = RCONST(0.000000001)*jstencil;
              zloc_value = RCONST(0.000000000001)*kstencil;
              true_value = myid_value + species_value
                + xloc_value + yloc_value + zloc_value;
              recv_value = w1d[l][v];
              if (abs(recv_value-true_value) > test_tol) {
                cout << "pack1D_y_bdry error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                     << v << ", " << i << ", " << udata.nyl << ", " << k << ", " << l << "), w1d = "
                     << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
                loc_errs++;
              }
            }
        }

        // z-directional stencil
        udata.pack1D_z_bdry(w1d, rho, mx, my, mz, et, chem, i, j, k);
        for (l=0; l<6; l++)
          for (v=0; v<NVAR; v++) {
            myid_value = RCONST(0.01)*udata.myid;
            if (k+l-3 < 0)            myid_value = RCONST(0.01)*udata.ipB;
            if (k+l-3 > udata.nzl-1)  myid_value = RCONST(0.01)*udata.ipF;
            species_value = RCONST(0.001)*v;
            istencil = i+udata.is;
            jstencil = j+udata.js;
            kstencil = k+l-3+udata.ks;
            if (kstencil < 0)           kstencil += udata.nz;
            if (kstencil > udata.nz-1)  kstencil -= udata.nz;
            xloc_value = RCONST(0.000001)*istencil;
            yloc_value = RCONST(0.000000001)*jstencil;
            zloc_value = RCONST(0.000000000001)*kstencil;
            true_value = myid_value + species_value
              + xloc_value + yloc_value + zloc_value;
            recv_value = w1d[l][v];
            if (abs(recv_value-true_value) > test_tol) {
              cout << "pack1D_z_bdry error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                   << v << ", " << i << ", " << j << ", " << k << ", " << l << "), w1d = "
                   << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
              loc_errs++;
            }
          }

        // z-directional stencil at upper boundary face
        if (k == udata.nzl-1) {
          udata.pack1D_z_bdry(w1d, rho, mx, my, mz, et, chem, i, j, udata.nzl);
          for (l=0; l<6; l++)
            for (v=0; v<NVAR; v++) {
              myid_value = RCONST(0.01)*udata.myid;
              if (udata.nzl+l-3 < 0)            myid_value = RCONST(0.01)*udata.ipB;
              if (udata.nzl+l-3 > udata.nzl-1)  myid_value = RCONST(0.01)*udata.ipF;
              species_value = RCONST(0.001)*v;
              istencil = i+udata.is;
              jstencil = j+udata.js;
              kstencil = udata.nzl+l-3+udata.ks;
              if (kstencil < 0)           kstencil += udata.nz;
              if (kstencil > udata.nz-1)  kstencil -= udata.nz;
              xloc_value = RCONST(0.000001)*istencil;
              yloc_value = RCONST(0.000000001)*jstencil;
              zloc_value = RCONST(0.000000000001)*kstencil;
              true_value = myid_value + species_value
                + xloc_value + yloc_value + zloc_value;
              recv_value = w1d[l][v];
              if (abs(recv_value-true_value) > test_tol) {
                cout << "pack1D_z_bdry error: myid = " << udata.myid << ", (v,i,j,k,l) = ("
                     << v << ", " << i << ", " << j << ", " << udata.nzl << ", " << l << "), w1d = "
                     << fixed << setprecision(13) << recv_value << " != " << true_value << endl;
                loc_errs++;
              }
            }
        }
      }


  // report on total errors encountered
  long int tot_errs = 0;
  retval = MPI_Reduce(&loc_errs, &tot_errs, 1, MPI_LONG, MPI_SUM, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (main)", 3)) return(1);
  if (udata.myid == 0)
    cout << "\n\nCommunication_test result: " << tot_errs << " total errors\n";

  // Clean up and return with successful completion
  N_VDestroy(w);               // Free solution vectors
  for (i=0; i<Nsubvecs; i++)
    N_VDestroy(wsubvecs[i]);
  delete[] wsubvecs;
  MPI_Finalize();              // Finalize MPI
  return 0;
}

//---- end of file ----
