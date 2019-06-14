/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Header file for shared main routine, utility routines and
 UserData class.
---------------------------------------------------------------*/

// Header files
#include <stdio.h>
#include <iostream>
#include <utility>
#include <string>
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


// Utility routine to check function return values
int check_flag(const void *flagvalue, const string funcname, const int opt);


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
  int npx;              // number of MPI processes in each direction
  int npy;
  int npz;
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
      nx(nx_), ny(ny_), nz(nz_), xl(xl_), xr(xr_), yl(yl_), yr(yr_), zl(zl_),
      zr(zr_), xlbc(xlbc_), xrbc(xrbc_), ylbc(ylbc_), yrbc(yrbc_), zlbc(zlbc_),
      zrbc(zrbc_), is(0), ie(0), js(0), je(0), ks(0), ke(0), nxl(0), nyl(0),
      nzl(0), comm(MPI_COMM_WORLD), myid(0), nprocs(0), npx(0), npy(0), npz(0),
      Erecv(NULL), Wrecv(NULL), Nrecv(NULL), Srecv(NULL), Frecv(NULL), Brecv(NULL),
      Esend(NULL), Wsend(NULL), Nsend(NULL), Ssend(NULL), Fsend(NULL), Bsend(NULL),
      ipW(-1), ipE(-1), ipS(-1), ipN(-1), ipB(-1), ipF(-1), gamma(gamma_)
  {
    dx = (xr-xl)/nx;
    dy = (yr-yl)/ny;
    dz = (zr-zl)/nz;
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

  // Set up parallel decomposition
  int SetupDecomp()
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
    npx = dims[0];
    npy = dims[1];
    npz = dims[2];

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
  int ExchangeStart(N_Vector w)
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
    realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
    if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;

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
            Wsend[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[IDX(i,j,k,nxl,nyl)];
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
            Esend[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[IDX(nxl-3+i,j,k,nxl,nyl)];
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
            Ssend[BUFIDX(4,i,j,k,nxl,3,nzl)] = et[IDX(i,j,k,nxl,nyl)];
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
            Nsend[BUFIDX(4,i,j,k,nxl,3,nzl)] = et[IDX(i,nyl-3+j,k,nxl,nyl)];
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
            Bsend[BUFIDX(4,i,j,k,nxl,nyl,3)] = et[IDX(i,j,k,nxl,nyl)];
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
            Fsend[BUFIDX(4,i,j,k,nxl,nyl,3)] = et[IDX(i,j,nzl-3+k,nxl,nyl)];
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
              Wrecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[IDX(2-i,j,k,nxl,nyl)];
            }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<nyl; j++)
            for (i=0; i<3; i++) {
              Wrecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = -rho[IDX(2-i,j,k,nxl,nyl)];
              Wrecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = -mx[IDX(2-i,j,k,nxl,nyl)];
              Wrecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = -my[IDX(2-i,j,k,nxl,nyl)];
              Wrecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = -mz[IDX(2-i,j,k,nxl,nyl)];
              Wrecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = -et[IDX(2-i,j,k,nxl,nyl)];
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
              Erecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[IDX(nxl-3+i,j,k,nxl,nyl)];
            }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<nyl; j++)
            for (i=0; i<3; i++) {
              Erecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = -rho[IDX(nxl-3+i,j,k,nxl,nyl)];
              Erecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = -mx[IDX(nxl-3+i,j,k,nxl,nyl)];
              Erecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = -my[IDX(nxl-3+i,j,k,nxl,nyl)];
              Erecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = -mz[IDX(nxl-3+i,j,k,nxl,nyl)];
              Erecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = -et[IDX(nxl-3+i,j,k,nxl,nyl)];
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
              Srecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = et[IDX(i,2-j,k,nxl,nyl)];
            }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<3; j++)
            for (i=0; i<nxl; i++) {
              Srecv[BUFIDX(0,i,j,k,nxl,3,nzl)] = -rho[IDX(i,2-j,k,nxl,nyl)];
              Srecv[BUFIDX(1,i,j,k,nxl,3,nzl)] = -mx[IDX(i,2-j,k,nxl,nyl)];
              Srecv[BUFIDX(2,i,j,k,nxl,3,nzl)] = -my[IDX(i,2-j,k,nxl,nyl)];
              Srecv[BUFIDX(3,i,j,k,nxl,3,nzl)] = -mz[IDX(i,2-j,k,nxl,nyl)];
              Srecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = -et[IDX(i,2-j,k,nxl,nyl)];
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
              Nrecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = et[IDX(i,nyl-3+j,k,nxl,nyl)];
            }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<3; j++)
            for (i=0; i<nxl; i++) {
              Nrecv[BUFIDX(0,i,j,k,nxl,3,nzl)] = -rho[IDX(i,nyl-3+j,k,nxl,nyl)];
              Nrecv[BUFIDX(1,i,j,k,nxl,3,nzl)] = -mx[IDX(i,nyl-3+j,k,nxl,nyl)];
              Nrecv[BUFIDX(2,i,j,k,nxl,3,nzl)] = -my[IDX(i,nyl-3+j,k,nxl,nyl)];
              Nrecv[BUFIDX(3,i,j,k,nxl,3,nzl)] = -mz[IDX(i,nyl-3+j,k,nxl,nyl)];
              Nrecv[BUFIDX(4,i,j,k,nxl,3,nzl)] = -et[IDX(i,nyl-3+j,k,nxl,nyl)];
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
              Brecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = et[IDX(i,j,2-k,nxl,nyl)];
            }
      } else {          // homogeneous Dirichlet
        for (k=0; k<3; k++)
          for (j=0; j<nyl; j++)
            for (i=0; i<nxl; i++) {
              Brecv[BUFIDX(0,i,j,k,nxl,nyl,3)] = -rho[IDX(i,j,2-k,nxl,nyl)];
              Brecv[BUFIDX(1,i,j,k,nxl,nyl,3)] = -mx[IDX(i,j,2-k,nxl,nyl)];
              Brecv[BUFIDX(2,i,j,k,nxl,nyl,3)] = -my[IDX(i,j,2-k,nxl,nyl)];
              Brecv[BUFIDX(3,i,j,k,nxl,nyl,3)] = -mz[IDX(i,j,2-k,nxl,nyl)];
              Brecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = -et[IDX(i,j,2-k,nxl,nyl)];
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
              Frecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = et[IDX(i,j,nzl-3+k,nxl,nyl)];
            }
      } else {          // homogeneous Dirichlet
        for (k=0; k<3; k++)
          for (j=0; j<nyl; j++)
            for (i=0; i<nxl; i++) {
              Frecv[BUFIDX(0,i,j,k,nxl,nyl,3)] = -rho[IDX(i,j,nzl-3+k,nxl,nyl)];
              Frecv[BUFIDX(1,i,j,k,nxl,nyl,3)] = -mx[IDX(i,j,nzl-3+k,nxl,nyl)];
              Frecv[BUFIDX(2,i,j,k,nxl,nyl,3)] = -my[IDX(i,j,nzl-3+k,nxl,nyl)];
              Frecv[BUFIDX(3,i,j,k,nxl,nyl,3)] = -mz[IDX(i,j,nzl-3+k,nxl,nyl)];
              Frecv[BUFIDX(4,i,j,k,nxl,nyl,3)] = -et[IDX(i,j,nzl-3+k,nxl,nyl)];
            }
      }
    }

    return 0;     // return with success flag
  }

  // Finish neighbor exchange
  int ExchangeEnd()
  {
    // local variables
    MPI_Status stat[12];
    int retval;

    // wait for messages to finish send/receive
    retval = MPI_Waitall(12, req, stat);
    if (check_flag(&retval, "MPI_Waitall (UserData::ExchangeEnd)", 3)) return -1;

    return 0;     // return with success flag
  }


};   // end UserData;




// Additional utility routines

//    Load inputs from file
int load_inputs(int myid, double& xl, double& xr, double& yl,
                double& yr, double& zl, double& zr, double& t0,
                double& tf, double& gamma, long int& nx, long int& ny,
                long int& nz, long int& xlbc, long int& xrbc,
                long int& ylbc, long int& yrbc, long int& zlbc,
                long int& zrbc, int& nout, int& showstats);

//    Equation of state
inline realtype eos(const realtype& rho, const realtype& mx,
                    const realtype& my, const realtype& mz,
                    const realtype& et, const UserData& udata);

//    Check for legal state
inline int legal_state(const realtype& rho, const realtype& mx,
                       const realtype& my, const realtype& mz,
                       const realtype& et, const UserData& udata);

//    Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);

//    Forcing terms
int external_forces(const realtype& t, N_Vector G, const UserData& udata);

//    Signed fluxes over patch
void signed_fluxes(const realtype (&w1d)[7][5], realtype (&fms)[7][5],
                   realtype (&fps)[7][5], const UserData& udata);

//    Print solution statistics
int print_stats(const realtype& t, const N_Vector w,
                const int& firstlast, const UserData& udata);

//    Output current solution
int output_solution(const N_Vector w, const int& newappend,
                    const UserData& udata);

//    1D packing routines
inline void pack1D_x(realtype (&w1d)[7][5], const realtype* rho,
                     const realtype* mx, const realtype* my,
                     const realtype* mz, const realtype* et,
                     const realtype* Wrecv, const realtype* Erecv,
                     const long int& i, const long int& j,
                     const long int& k, const long int& nxl,
                     const long int& nyl, const long int& nzl);
inline void pack1D_y(realtype (&w1d)[7][5], const realtype* rho,
                     const realtype* mx, const realtype* my,
                     const realtype* mz, const realtype* et,
                     const realtype* Srecv, const realtype* Nrecv,
                     const long int& i, const long int& j,
                     const long int& k, const long int& nxl,
                     const long int& nyl, const long int& nzl);
inline void pack1D_z(realtype (&w1d)[7][5], const realtype* rho,
                     const realtype* mx, const realtype* my,
                     const realtype* mz, const realtype* et,
                     const realtype* Brecv, const realtype* Frecv,
                     const long int& i, const long int& j,
                     const long int& k, const long int& nxl,
                     const long int& nyl, const long int& nzl);

//    WENO Div(flux(u)) function
void div_flux(realtype (&w1d)[7][5], const int& idir,
              const realtype& dx, realtype* dw, const UserData& udata);

//    Parameter input helper function
void* arkstep_init_from_file(const char fname[], const ARKRhsFn f,
                             const ARKRhsFn fe, const ARKRhsFn fi,
                             const realtype t0, const N_Vector w0,
                             int& imex, int& dense_order, int& fxpt,
                             double& rtol, double& atol);

//---- end of file ----
