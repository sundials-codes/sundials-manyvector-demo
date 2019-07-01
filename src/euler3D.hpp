/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Header file for shared main routine, utility routines, 
 ARKodeParameters and UserData class.
 ---------------------------------------------------------------*/

// only include this file once (if included multiple times)
#ifndef __EULER3D_HPP__
#define __EULER3D_HPP__

// Header files
#include <stdio.h>
#include <iostream>
#include <utility>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <arkode/arkode_arkstep.h>
#include <nvector/nvector_mpimanyvector.h>
#include <nvector/nvector_manyvector.h>
#include <nvector/nvector_parallel.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <mpi.h>
#include <exception>

using namespace std;

// define default value for total number of variables per spatial location (if undefined)
#ifndef NVAR
#define NVAR 5
#endif

// ensure that NVAR is no smaller than 5
#if NVAR < 5
#error *** Illegal NVAR value (must be at least 5) ***
#endif

// accessor macro between (i,j,k) location and 1D data array location
#define IDX(i,j,k,nx,ny,nz) ( (i) + (nx)*((j) + (ny)*(k)) )

// accessor macro between (v,i,j,k) location and 1D data array location
#define BUFIDX(v,i,j,k,nx,ny,nz) ( (v) + (NVAR)*((i) + (nx)*((j) + (ny)*(k))) )


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


// Utility routine to check function return values:
int check_flag(const void *flagvalue, const string funcname, const int opt);


// ARKode solver parameters class
class ARKodeParameters {

public:
  int order;           // temporal order of accuracy
  int dense_order;     // dense output order of accuracy (<0 => no interpolation)
  int btable;          // specific built-in Butcher table to use (<0 => default)
  int adapt_method;    // temporal adaptivity algorithm to use
  int maxnef;          // max num temporal error failures (0 => default)
  int mxhnil;          // max num tn+h=tn warnings (0 => default)
  int mxsteps;         // max internal steps per 'evolve' call (0 => default)
  double cflfac;       // max multiple of dt_cfl condition to use (0 => default)
  double safety;       // temporal step size safety factor (0 => default)
  double bias;         // temporal error bias factor (0 => default)
  double growth;       // max temporal growth/step (0 => default)
  int pq;              // use of method/embedding order for adaptivity (0 => default)
  double k1;           // temporal adaptivity parameters (0 => default)
  double k2;
  double k3;
  double etamx1;       // max change after first internal step (0 => default)
  double etamxf;       // max change on a general internal step (0 => default)
  double h0;           // initial time step size (0 => default)
  double hmin;         // minimum time step size (0 => default)
  double hmax;         // maximum time step size (0 => infinite)
  double rtol;         // relative solution tolerance (0 => default)
  double atol;         // absolute solution tolerance (0 => default)
  
  // constructor (with default values)
  ARKodeParameters() :
    order(4), dense_order(-1), btable(-1), adapt_method(0), maxnef(0),
    mxhnil(0), mxsteps(5000), cflfac(0.0), safety(0.0), bias(0.0),
    growth(0.0), pq(0), k1(0.0), k2(0.0), k3(0.0), etamx1(0.0),
    etamxf(0.0), h0(0.0), hmin(0.0), hmax(0.0), rtol(1e-8), atol(1e-12)
  {};

};   // end ARKodeParameters;



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
  realtype t0;
  realtype tf;
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
  int      nchem;       // number of tracers/chemical species
  realtype gamma;       // ratio of specific heat capacities, cp/cv
  realtype cfl;         // fraction of maximum stable step size to use

  ///// run-control parameters /////
  int nout;             // num pauses in integration to run diagnostics/io
  int showstats;        // flag indicating whether to display run stats to screen
  
  ///// reusable arrays for WENO flux calculations /////
  realtype *xflux;
  realtype *yflux;
  realtype *zflux;
  
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
  UserData() :
      nx(3), ny(3), nz(3), xl(0.0), xr(1.0), yl(0.0), yr(1.0), zl(0.0),
      zr(1.0), t0(0.0), tf(1.0), xlbc(0), xrbc(0), ylbc(0), yrbc(0),
      zlbc(0), zrbc(0), is(-1), ie(-1), js(-1), je(-1), ks(-1), ke(-1),
      nxl(-1), nyl(-1), nzl(-1), comm(MPI_COMM_WORLD), myid(-1),
      nprocs(-1), npx(-1), npy(-1), npz(-1), Erecv(NULL), Wrecv(NULL), Nrecv(NULL),
      Srecv(NULL), Frecv(NULL), Brecv(NULL), Esend(NULL), Wsend(NULL), Nsend(NULL),
      Ssend(NULL), Fsend(NULL), Bsend(NULL), ipW(-1), ipE(-1), ipS(-1), ipN(-1),
      ipB(-1), ipF(-1), gamma(1.4), cfl(0.0), xflux(NULL), yflux(NULL),
      zflux(NULL), nout(10), showstats(0)
  {
    nchem = (NVAR) - 5;
  };
  // destructor
  ~UserData() {
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
    if (xflux != NULL)  delete[] xflux;
    if (yflux != NULL)  delete[] yflux;
    if (zflux != NULL)  delete[] zflux;
  };

  // Set up parallel decomposition
  int SetupDecomp()
  {
    // local variables
    int i, retval, truedims, periods[3], coords[3], nbcoords[3];
    int dims_suggested[] = {0, 0, 0};
    int dims[] = {1, 1, 1};

    // check that this has not been called before
    if (Erecv != NULL || Wrecv != NULL || Srecv != NULL ||
        Nrecv != NULL || Brecv != NULL || Frecv != NULL ||
        xflux != NULL || yflux != NULL || zflux != NULL) {
      cerr << "SetupDecomp warning: parallel decomposition already set up\n";
      return 1;
    }

    // set mesh sizes
    dx = (xr-xl)/nx;
    dy = (yr-yl)/ny;
    dz = (zr-zl)/nz;

    // determine "true" problem dimensionality; since the minimum domain size in any
    // given direction is 3, only parallelize dimensions with more than 3 unknowns
    truedims = 0;
    truedims += (nx > 3) ? 1 : 0;
    truedims += (ny > 3) ? 1 : 0;
    truedims += (nz > 3) ? 1 : 0;
    
    // get suggested parallel decomposition
    retval = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (check_flag(&retval, "MPI_Comm_size (UserData::SetupDecomp)", 3)) return -1;
    retval = MPI_Dims_create(nprocs, truedims, dims_suggested);
    if (check_flag(&retval, "MPI_Dims_create (UserData::SetupDecomp)", 3)) return -1;

    // adjust dims to match actual problem dimensions
    i=0;
    if (nx > 3) 
      dims[0] = dims_suggested[i++];
    if (ny > 3) 
      dims[1] = dims_suggested[i++];
    if (nz > 3) 
      dims[2] = dims_suggested[i++];
    
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

    // ensure that each dimension has a local extent at least 3
    if (nxl < 3) {
      cerr << "SetupDecomp error: task " << myid << " has nxl = " << nxl << " < 3\n";
      return(-1);
    }
    if (nyl < 3) {
      cerr << "SetupDecomp error: task " << myid << " has nyl = " << nyl << " < 3\n";
      return(-1);
    }
    if (nzl < 3) {
      cerr << "SetupDecomp error: task " << myid << " has nzl = " << nzl << " < 3\n";
      return(-1);
    }
    
    // allocate temporary arrays for storing directional fluxes
    xflux = new realtype[(NVAR)*(nxl+1)*nyl*nzl];
    yflux = new realtype[(NVAR)*nxl*(nyl+1)*nzl];
    zflux = new realtype[(NVAR)*nxl*nyl*(nzl+1)];
    
    // for all faces where neighbors exist: determine neighbor process indices;
    // for all faces: allocate exchange buffers (external boundaries fill with ghost values)
    Wrecv = new realtype[(NVAR)*3*nyl*nzl];
    Wsend = new realtype[(NVAR)*3*nyl*nzl];
    ipW = MPI_PROC_NULL;
    if ((coords[0] > 0) || (xlbc == 0)) {
      nbcoords[0] = coords[0]-1;
      nbcoords[1] = coords[1];
      nbcoords[2] = coords[2];
      retval = MPI_Cart_rank(comm, nbcoords, &ipW);
      if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
    }

    Erecv = new realtype[(NVAR)*3*nyl*nzl];
    Esend = new realtype[(NVAR)*3*nyl*nzl];
    ipE = MPI_PROC_NULL;
    if ((coords[0] < dims[0]-1) || (xrbc == 0)) {
      nbcoords[0] = coords[0]+1;
      nbcoords[1] = coords[1];
      nbcoords[2] = coords[2];
      retval = MPI_Cart_rank(comm, nbcoords, &ipE);
      if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
    }

    Srecv = new realtype[(NVAR)*nxl*3*nzl];
    Ssend = new realtype[(NVAR)*nxl*3*nzl];
    ipS = MPI_PROC_NULL;
    if ((coords[1] > 0) || (ylbc == 0)) {
      nbcoords[0] = coords[0];
      nbcoords[1] = coords[1]-1;
      nbcoords[2] = coords[2];
      retval = MPI_Cart_rank(comm, nbcoords, &ipS);
      if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
    }

    Nrecv = new realtype[(NVAR)*nxl*3*nzl];
    Nsend = new realtype[(NVAR)*nxl*3*nzl];
    ipN = MPI_PROC_NULL;
    if ((coords[1] < dims[1]-1) || (yrbc == 0)) {
      nbcoords[0] = coords[0];
      nbcoords[1] = coords[1]+1;
      nbcoords[2] = coords[2];
      retval = MPI_Cart_rank(comm, nbcoords, &ipN);
      if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
    }

    Brecv = new realtype[(NVAR)*nxl*nyl*3];
    Bsend = new realtype[(NVAR)*nxl*nyl*3];
    ipB = MPI_PROC_NULL;
    if ((coords[2] > 0) || (zlbc == 0)) {
      nbcoords[0] = coords[0];
      nbcoords[1] = coords[1];
      nbcoords[2] = coords[2]-1;
      retval = MPI_Cart_rank(comm, nbcoords, &ipB);
      if (check_flag(&retval, "MPI_Cart_rank (UserData::SetupDecomp)", 3)) return -1;
    }

    Frecv = new realtype[(NVAR)*nxl*nyl*3];
    Fsend = new realtype[(NVAR)*nxl*nyl*3];
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

    // access data arrays
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
    realtype *chem;
    
    // initialize all requests in array
    for (i=0; i<12; i++)  req[i] = MPI_REQUEST_NULL;

    // open an Irecv buffer for each neighbor
    if (ipW != MPI_PROC_NULL) {
      retval = MPI_Irecv(Wrecv, (NVAR)*3*nyl*nzl, MPI_SUNREALTYPE, ipW,
                         1, comm, req);
      if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipE != MPI_PROC_NULL) {
      retval = MPI_Irecv(Erecv, (NVAR)*3*nyl*nzl, MPI_SUNREALTYPE, ipE,
                         0, comm, req+1);
      if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipS != MPI_PROC_NULL) {
      retval = MPI_Irecv(Srecv, (NVAR)*nxl*3*nzl, MPI_SUNREALTYPE, ipS,
                         3, comm, req+2);
      if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipN != MPI_PROC_NULL) {
      retval = MPI_Irecv(Nrecv, (NVAR)*nxl*3*nzl, MPI_SUNREALTYPE, ipN,
                         2, comm, req+3);
      if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipB != MPI_PROC_NULL) {
      retval = MPI_Irecv(Brecv, (NVAR)*nxl*nyl*3, MPI_SUNREALTYPE, ipB,
                         5, comm, req+4);
      if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipF != MPI_PROC_NULL) {
      retval = MPI_Irecv(Frecv, (NVAR)*nxl*nyl*3, MPI_SUNREALTYPE, ipF,
                         4, comm, req+5);
      if (check_flag(&retval, "MPI_Irecv (UserData::ExchangeStart)", 3)) return -1;
    }

    // send data to neighbors
    if (ipW != MPI_PROC_NULL) {
      for (k=0; k<nzl; k++)
        for (j=0; j<nyl; j++) {
          for (i=0; i<3; i++) 
            Wsend[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++) 
            Wsend[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++) 
            Wsend[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++) 
            Wsend[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++) 
            Wsend[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[ IDX(i,j,k,nxl,nyl,nzl)];
          if (nchem>0) {
            for (i=0; i<3; i++) {
              chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,k,nxl,nyl,nzl));
              if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
              for (v=0; v<nchem; v++)
                Wsend[BUFIDX(5+v,i,j,k,3,nyl,nzl)] = chem[v];
            }
          }
        }
      retval = MPI_Isend(Wsend, (NVAR)*3*nyl*nzl, MPI_SUNREALTYPE, ipW, 0,
                         comm, req+6);
      if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipE != MPI_PROC_NULL) {
      for (k=0; k<nzl; k++)
        for (j=0; j<nyl; j++) {
          for (i=0; i<3; i++)
            Esend[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++)
            Esend[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++)
            Esend[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++)
            Esend[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
          for (i=0; i<3; i++)
            Esend[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
          if (nchem>0) {
            for (i=0; i<3; i++) {
              chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(nxl-3+i,j,k,nxl,nyl,nzl));
              if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
              for (v=0; v<nchem; v++)
                Esend[BUFIDX(5+v,i,j,k,3,nyl,nzl)] = chem[v];
            }
          }
        }
      retval = MPI_Isend(Esend, (NVAR)*3*nyl*nzl, MPI_SUNREALTYPE, ipE, 1,
                         comm, req+7);
      if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipS != MPI_PROC_NULL) {
      for (k=0; k<nzl; k++)
        for (j=0; j<3; j++) {
          for (i=0; i<nxl; i++) 
            Ssend[BUFIDX(0,j,i,k,3,nxl,nzl)] = rho[IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Ssend[BUFIDX(1,j,i,k,3,nxl,nzl)] = mx[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Ssend[BUFIDX(2,j,i,k,3,nxl,nzl)] = my[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Ssend[BUFIDX(3,j,i,k,3,nxl,nzl)] = mz[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Ssend[BUFIDX(4,j,i,k,3,nxl,nzl)] = et[ IDX(i,j,k,nxl,nyl,nzl)];
          if (nchem>0) {
            for (i=0; i<nxl; i++) {
              chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,k,nxl,nyl,nzl));
              if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
              for (v=0; v<nchem; v++)
                Ssend[BUFIDX(5+v,j,i,k,3,nxl,nzl)] = chem[v];
            }
          }
        }
      retval = MPI_Isend(Ssend, (NVAR)*nxl*3*nzl, MPI_SUNREALTYPE, ipS, 2,
                         comm, req+8);
      if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipN != MPI_PROC_NULL) {
      for (k=0; k<nzl; k++)
        for (j=0; j<3; j++) {
          for (i=0; i<nxl; i++) 
            Nsend[BUFIDX(0,j,i,k,3,nxl,nzl)] = rho[IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Nsend[BUFIDX(1,j,i,k,3,nxl,nzl)] = mx[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Nsend[BUFIDX(2,j,i,k,3,nxl,nzl)] = my[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Nsend[BUFIDX(3,j,i,k,3,nxl,nzl)] = mz[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Nsend[BUFIDX(4,j,i,k,3,nxl,nzl)] = et[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
          if (nchem>0) {
            for (i=0; i<nxl; i++) {
              chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,nyl-3+j,k,nxl,nyl,nzl));
              if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
              for (v=0; v<nchem; v++)
                Nsend[BUFIDX(5+v,j,i,k,3,nxl,nzl)] = chem[v];
            }
          }
        }
      retval = MPI_Isend(Nsend, (NVAR)*nxl*3*nzl, MPI_SUNREALTYPE, ipN, 3,
                         comm, req+9);
      if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipB != MPI_PROC_NULL) {
      for (k=0; k<3; k++)
        for (j=0; j<nyl; j++) {
          for (i=0; i<nxl; i++) 
            Bsend[BUFIDX(0,k,i,j,3,nxl,nyl)] = rho[IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Bsend[BUFIDX(1,k,i,j,3,nxl,nyl)] = mx[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Bsend[BUFIDX(2,k,i,j,3,nxl,nyl)] = my[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Bsend[BUFIDX(3,k,i,j,3,nxl,nyl)] = mz[ IDX(i,j,k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Bsend[BUFIDX(4,k,i,j,3,nxl,nyl)] = et[ IDX(i,j,k,nxl,nyl,nzl)];
          if (nchem>0) {
            for (i=0; i<nxl; i++) {
              chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,k,nxl,nyl,nzl));
              if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
              for (v=0; v<nchem; v++)
                Bsend[BUFIDX(5+v,k,i,j,3,nxl,nyl)] = chem[v];
            }
          }
        }
      retval = MPI_Isend(Bsend, (NVAR)*nxl*nyl*3, MPI_SUNREALTYPE, ipB, 4,
                         comm, req+10);
      if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
    }

    if (ipF != MPI_PROC_NULL) {
      for (k=0; k<3; k++)
        for (j=0; j<nyl; j++) {
          for (i=0; i<nxl; i++) 
            Fsend[BUFIDX(0,k,i,j,3,nxl,nyl)] = rho[IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Fsend[BUFIDX(1,k,i,j,3,nxl,nyl)] = mx[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Fsend[BUFIDX(2,k,i,j,3,nxl,nyl)] = my[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Fsend[BUFIDX(3,k,i,j,3,nxl,nyl)] = mz[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
          for (i=0; i<nxl; i++) 
            Fsend[BUFIDX(4,k,i,j,3,nxl,nyl)] = et[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
          if (nchem>0) {
            for (i=0; i<nxl; i++) {
              chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,nzl-3+k,nxl,nyl,nzl));
              if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
              for (v=0; v<nchem; v++)
                Fsend[BUFIDX(5+v,k,i,j,3,nxl,nyl)] = chem[v];
            }
          }
        }
      retval = MPI_Isend(Fsend, (NVAR)*nxl*nyl*3, MPI_SUNREALTYPE, ipF, 5,
                         comm, req+11);
      if (check_flag(&retval, "MPI_Isend (UserData::ExchangeStart)", 3)) return -1;
    }

    // if this process owns any external faces, fill ghost zones
    // to satisfy homogeneous Neumann boundary conditions

    //     West face
    if (ipW == MPI_PROC_NULL) {
      if (xlbc == 1) {  // homogeneous Neumann
        for (k=0; k<nzl; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[ IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[ IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[ IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[ IDX(2-i,j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<3; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(2-i,j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Wrecv[BUFIDX(5+v,i,j,k,3,nyl,nzl)] = chem[v];
              }
            }
          }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = -rho[IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = -mx[ IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = -my[ IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = -mz[ IDX(2-i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Wrecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = -et[ IDX(2-i,j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<3; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(2-i,j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Wrecv[BUFIDX(5+v,i,j,k,3,nyl,nzl)] = -chem[v];
              }
            }
          }
      }
    }

    //     East face
    if (ipE == MPI_PROC_NULL) {
      if (xrbc == 1) {  // homogeneous Neumann
        for (k=0; k<nzl; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = rho[IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = mx[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = my[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = mz[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = et[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<3; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(nxl-3+i,j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Erecv[BUFIDX(5+v,i,j,k,3,nyl,nzl)] = chem[v];
              }
            }
          }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(0,i,j,k,3,nyl,nzl)] = -rho[IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(1,i,j,k,3,nyl,nzl)] = -mx[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(2,i,j,k,3,nyl,nzl)] = -my[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(3,i,j,k,3,nyl,nzl)] = -mz[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            for (i=0; i<3; i++) 
              Erecv[BUFIDX(4,i,j,k,3,nyl,nzl)] = -et[ IDX(nxl-3+i,j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<3; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(nxl-3+i,j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Erecv[BUFIDX(5+v,i,j,k,3,nyl,nzl)] = -chem[v];
              }
            }
          }
      }
    }

    //     South face
    if (ipS == MPI_PROC_NULL) {
      if (ylbc == 1) {  // homogeneous Neumann
        for (k=0; k<nzl; k++)
          for (j=0; j<3; j++) {
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(0,j,i,k,3,nxl,nzl)] = rho[IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(1,j,i,k,3,nxl,nzl)] = mx[ IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(2,j,i,k,3,nxl,nzl)] = my[ IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(3,j,i,k,3,nxl,nzl)] = mz[ IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(4,j,i,k,3,nxl,nzl)] = et[ IDX(i,2-j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,2-j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Srecv[BUFIDX(5+v,j,i,k,3,nxl,nzl)] = chem[v];
              }
            }
          }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<3; j++) {
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(0,j,i,k,3,nxl,nzl)] = -rho[IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(1,j,i,k,3,nxl,nzl)] = -mx[ IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(2,j,i,k,3,nxl,nzl)] = -my[ IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(3,j,i,k,3,nxl,nzl)] = -mz[ IDX(i,2-j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Srecv[BUFIDX(4,j,i,k,3,nxl,nzl)] = -et[ IDX(i,2-j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,2-j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Srecv[BUFIDX(5+v,j,i,k,3,nxl,nzl)] = -chem[v];
              }
            }
          }
      }
    }

    //     North face
    if (ipN == MPI_PROC_NULL) {
      if (yrbc == 1) {  // homogeneous Neumann
        for (k=0; k<nzl; k++)
          for (j=0; j<3; j++) {
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(0,j,i,k,3,nxl,nzl)] = rho[IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(1,j,i,k,3,nxl,nzl)] = mx[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(2,j,i,k,3,nxl,nzl)] = my[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(3,j,i,k,3,nxl,nzl)] = mz[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(4,j,i,k,3,nxl,nzl)] = et[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,nyl-3+j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Nrecv[BUFIDX(5+v,j,i,k,3,nxl,nzl)] = chem[v];
              }
            }
          }
      } else {          // homogeneous Dirichlet
        for (k=0; k<nzl; k++)
          for (j=0; j<3; j++) {
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(0,j,i,k,3,nxl,nzl)] = -rho[IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(1,j,i,k,3,nxl,nzl)] = -mx[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(2,j,i,k,3,nxl,nzl)] = -my[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(3,j,i,k,3,nxl,nzl)] = -mz[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Nrecv[BUFIDX(4,j,i,k,3,nxl,nzl)] = -et[ IDX(i,nyl-3+j,k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,nyl-3+j,k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Nrecv[BUFIDX(5+v,j,i,k,3,nxl,nzl)] = -chem[v];
              }
            }
          }
      }
    }

    //     Back face
    if (ipB == MPI_PROC_NULL) {
      if (zlbc == 1) {  // homogeneous Neumann
        for (k=0; k<3; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(0,k,i,j,3,nxl,nyl)] = rho[IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(1,k,i,j,3,nxl,nyl)] = mx[ IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(2,k,i,j,3,nxl,nyl)] = my[ IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(3,k,i,j,3,nxl,nyl)] = mz[ IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(4,k,i,j,3,nxl,nyl)] = et[ IDX(i,j,2-k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,2-k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Brecv[BUFIDX(5+v,k,i,j,3,nxl,nyl)] = chem[v];
              }
            }
          }
      } else {          // homogeneous Dirichlet
        for (k=0; k<3; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(0,k,i,j,3,nxl,nyl)] = -rho[IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(1,k,i,j,3,nxl,nyl)] = -mx[ IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(2,k,i,j,3,nxl,nyl)] = -my[ IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(3,k,i,j,3,nxl,nyl)] = -mz[ IDX(i,j,2-k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Brecv[BUFIDX(4,k,i,j,3,nxl,nyl)] = -et[ IDX(i,j,2-k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,2-k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Brecv[BUFIDX(5+v,k,i,j,3,nxl,nyl)] = -chem[v];
              }
            }
          }
      }
    }

    //     Front face
    if (ipF == MPI_PROC_NULL) {
      if (zrbc == 1) {  // homogeneous Neumann
        for (k=0; k<3; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(0,k,i,j,3,nxl,nyl)] = rho[IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(1,k,i,j,3,nxl,nyl)] = mx[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(2,k,i,j,3,nxl,nyl)] = my[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(3,k,i,j,3,nxl,nyl)] = mz[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(4,k,i,j,3,nxl,nyl)] = et[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,nzl-3+k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Frecv[BUFIDX(5+v,k,i,j,3,nxl,nyl)] = chem[v];
              }
            }
          }
      } else {          // homogeneous Dirichlet
        for (k=0; k<3; k++)
          for (j=0; j<nyl; j++) {
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(0,k,i,j,3,nxl,nyl)] = -rho[IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(1,k,i,j,3,nxl,nyl)] = -mx[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(2,k,i,j,3,nxl,nyl)] = -my[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(3,k,i,j,3,nxl,nyl)] = -mz[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            for (i=0; i<nxl; i++) 
              Frecv[BUFIDX(4,k,i,j,3,nxl,nyl)] = -et[ IDX(i,j,nzl-3+k,nxl,nyl,nzl)];
            if (nchem>0) {
              for (i=0; i<nxl; i++) {
                chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,nzl-3+k,nxl,nyl,nzl));
                if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (ExchangeStart)", 0)) return -1;
                for (v=0; v<nchem; v++)
                  Frecv[BUFIDX(5+v,k,i,j,3,nxl,nyl)] = -chem[v];
              }
            }
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

  // Utility routines to pack 1-dimensional data for *interior only* data;
  // e.g., in the x-direction given an (i,j,k) location, we return values at
  // the 6 nodal values closest to the (i-1/2,j,k) face along the x-direction, 
  // {w(i-3,j,k), w(i-2,j,k), w(i-1,j,k), w(i,j,k), w(i+1,j,k), w(i+2,j,k)}.
  inline void pack1D_x(realtype (&w1d)[6][NVAR], const realtype* rho,
                       const realtype* mx, const realtype* my,
                       const realtype* mz, const realtype* et,
                       const N_Vector w, const long int& i,
                       const long int& j, const long int& k) const
  {
    for (int l=0; l<6; l++)  w1d[l][0] = rho[IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][1] = mx[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][2] = my[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][3] = mz[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][4] = et[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    if (nchem>0) {
      for (int l=0; l<6; l++) {
        realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i-3+l,j,k,nxl,nyl,nzl));
        if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_x)", 0)) throw std::exception();
        for (int v=0; v<nchem; v++)  w1d[l][5+v] = chem[v];
      }
    }
  }
  inline void pack1D_y(realtype (&w1d)[6][NVAR], const realtype* rho,
                       const realtype* mx, const realtype* my,
                       const realtype* mz, const realtype* et,
                       const N_Vector w, const long int& i,
                       const long int& j, const long int& k) const
  {
    for (int l=0; l<6; l++)  w1d[l][0] = rho[IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][1] = mx[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][2] = my[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][3] = mz[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][4] = et[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    if (nchem>0) {
      for (int l=0; l<6; l++) {
        realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j-3+l,k,nxl,nyl,nzl));
        if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_y)", 0)) throw std::exception();
        for (int v=0; v<nchem; v++)  w1d[l][5+v] = chem[v];
      }
    }
  }
  inline void pack1D_z(realtype (&w1d)[6][NVAR], const realtype* rho,
                       const realtype* mx, const realtype* my,
                       const realtype* mz, const realtype* et,
                       const N_Vector w, const long int& i,
                       const long int& j, const long int& k) const
  {
    for (int l=0; l<6; l++)  w1d[l][0] = rho[IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][1] = mx[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][2] = my[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][3] = mz[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<6; l++)  w1d[l][4] = et[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    if (nchem>0) {
      for (int l=0; l<6; l++) {
        realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,k-3+l,nxl,nyl,nzl));
        if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_z)", 0)) throw std::exception();
        for (int v=0; v<nchem; v++)  w1d[l][5+v] = chem[v];
      }
    }
  }

  // Utility routines to pack 1-dimensional data for locations near the
  // subprocessor boundary; like the routines above these pack the 6 closest
  // entries aligned with, e.g., the (i-1/2,j,k) face, but now some entries
  // will come from receive buffers.
  inline void pack1D_x_bdry(realtype (&w1d)[6][NVAR], const realtype* rho,
                            const realtype* mx, const realtype* my,
                            const realtype* mz, const realtype* et,
                            const N_Vector w, const long int& i,
                            const long int& j, const long int& k) const
  {
    for (int l=0; l<3; l++) 
      w1d[l][0] = (i<(3-l)) ? Wrecv[BUFIDX(0,i+l,j,k,3,nyl,nzl)] : rho[IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l][1] = (i<(3-l)) ? Wrecv[BUFIDX(1,i+l,j,k,3,nyl,nzl)] : mx[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l][2] = (i<(3-l)) ? Wrecv[BUFIDX(2,i+l,j,k,3,nyl,nzl)] : my[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l][3] = (i<(3-l)) ? Wrecv[BUFIDX(3,i+l,j,k,3,nyl,nzl)] : mz[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l][4] = (i<(3-l)) ? Wrecv[BUFIDX(4,i+l,j,k,3,nyl,nzl)] : et[ IDX(i-3+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l+3][0] = (i>(nxl-l-1)) ? Erecv[BUFIDX(0,i-nxl+l,j,k,3,nyl,nzl)] : rho[IDX(i+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l+3][1] = (i>(nxl-l-1)) ? Erecv[BUFIDX(1,i-nxl+l,j,k,3,nyl,nzl)] : mx[ IDX(i+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l+3][2] = (i>(nxl-l-1)) ? Erecv[BUFIDX(2,i-nxl+l,j,k,3,nyl,nzl)] : my[ IDX(i+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l+3][3] = (i>(nxl-l-1)) ? Erecv[BUFIDX(3,i-nxl+l,j,k,3,nyl,nzl)] : mz[ IDX(i+l,j,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++) 
      w1d[l+3][4] = (i>(nxl-l-1)) ? Erecv[BUFIDX(4,i-nxl+l,j,k,3,nyl,nzl)] : et[ IDX(i+l,j,k,nxl,nyl,nzl)];
    if (nchem>0) {
      for (int l=0; l<3; l++) {
        if (i<(3-l)) {
          for (int v=0; v<nchem; v++)   w1d[l][5+v] = Wrecv[BUFIDX(5+v,i+l,j,k,3,nyl,nzl)];
        } else {
          realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i-3+l,j,k,nxl,nyl,nzl));
          if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_x_bdry)", 0)) throw std::exception();
          for (int v=0; v<nchem; v++)   w1d[l][5+v] = chem[v];
        }
        for (int v=0; v<nchem; v++)  w1d[l][5+v] = ZERO;
      }
      for (int l=0; l<3; l++) {
        if (i>(nxl-l-1)) {
          for (int v=0; v<nchem; v++)   w1d[l+3][5+v] = Erecv[BUFIDX(5+v,i-nxl+l,j,k,3,nyl,nzl)];
        } else {
          realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i+l,j,k,nxl,nyl,nzl));
          if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_x_bdry)", 0)) throw std::exception();
          for (int v=0; v<nchem; v++)   w1d[l+3][5+v] = chem[v];
        }
      }
    }
  }
  inline void pack1D_y_bdry(realtype (&w1d)[6][NVAR], const realtype* rho,
                            const realtype* mx, const realtype* my,
                            const realtype* mz, const realtype* et,
                            const N_Vector w, const long int& i,
                            const long int& j, const long int& k) const
  {
    for (int l=0; l<3; l++)
      w1d[l][0] = (j<(3-l)) ? Srecv[BUFIDX(0,j+l,i,k,3,nxl,nzl)] : rho[IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][1] = (j<(3-l)) ? Srecv[BUFIDX(1,j+l,i,k,3,nxl,nzl)] : mx[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][2] = (j<(3-l)) ? Srecv[BUFIDX(2,j+l,i,k,3,nxl,nzl)] : my[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][3] = (j<(3-l)) ? Srecv[BUFIDX(3,j+l,i,k,3,nxl,nzl)] : mz[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][4] = (j<(3-l)) ? Srecv[BUFIDX(4,j+l,i,k,3,nxl,nzl)] : et[ IDX(i,j-3+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][0] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(0,j-nyl+l,i,k,3,nxl,nzl)] : rho[IDX(i,j+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][1] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(1,j-nyl+l,i,k,3,nxl,nzl)] : mx[ IDX(i,j+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][2] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(2,j-nyl+l,i,k,3,nxl,nzl)] : my[ IDX(i,j+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][3] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(3,j-nyl+l,i,k,3,nxl,nzl)] : mz[ IDX(i,j+l,k,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][4] = (j>(nyl-l-1)) ? Nrecv[BUFIDX(4,j-nyl+l,i,k,3,nxl,nzl)] : et[ IDX(i,j+l,k,nxl,nyl,nzl)];
    if (nchem>0) {
      for (int l=0; l<3; l++) {
        if (j<(3-l)) {
          for (int v=0; v<nchem; v++)  w1d[l][5+v] = Srecv[BUFIDX(5+v,j+l,i,k,3,nxl,nzl)];
        } else {
          realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j-3+l,k,nxl,nyl,nzl));
          if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_y_bdry)", 0)) throw std::exception();
          for (int v=0; v<nchem; v++)  w1d[l][5+v] = chem[v];
        }
      }
      for (int l=0; l<3; l++) {
        if (j>(nyl-l-1)) {
          for (int v=0; v<nchem; v++)  w1d[l+3][5+v] = Nrecv[BUFIDX(5+v,j-nyl+l,i,k,3,nxl,nzl)];
        } else {
          realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j+l,k,nxl,nyl,nzl));
          if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_y_bdry)", 0)) throw std::exception();
          for (int v=0; v<nchem; v++)  w1d[l+3][5+v] = chem[v];
        }
      }
    }
  }
  inline void pack1D_z_bdry(realtype (&w1d)[6][NVAR], const realtype* rho,
                            const realtype* mx, const realtype* my,
                            const realtype* mz, const realtype* et,
                            const N_Vector w, const long int& i,
                            const long int& j, const long int& k) const
  {
    for (int l=0; l<3; l++)
      w1d[l][0] = (k<(3-l)) ? Brecv[BUFIDX(0,k+l,i,j,3,nxl,nyl)] : rho[IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][1] = (k<(3-l)) ? Brecv[BUFIDX(1,k+l,i,j,3,nxl,nyl)] : mx[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][2] = (k<(3-l)) ? Brecv[BUFIDX(2,k+l,i,j,3,nxl,nyl)] : my[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][3] = (k<(3-l)) ? Brecv[BUFIDX(3,k+l,i,j,3,nxl,nyl)] : mz[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l][4] = (k<(3-l)) ? Brecv[BUFIDX(4,k+l,i,j,3,nxl,nyl)] : et[ IDX(i,j,k-3+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][0] = (k>(nzl-l-1)) ? Frecv[BUFIDX(0,k-nzl+l,i,j,3,nxl,nyl)] : rho[IDX(i,j,k+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][1] = (k>(nzl-l-1)) ? Frecv[BUFIDX(1,k-nzl+l,i,j,3,nxl,nyl)] : mx[ IDX(i,j,k+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][2] = (k>(nzl-l-1)) ? Frecv[BUFIDX(2,k-nzl+l,i,j,3,nxl,nyl)] : my[ IDX(i,j,k+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][3] = (k>(nzl-l-1)) ? Frecv[BUFIDX(3,k-nzl+l,i,j,3,nxl,nyl)] : mz[ IDX(i,j,k+l,nxl,nyl,nzl)];
    for (int l=0; l<3; l++)
      w1d[l+3][4] = (k>(nzl-l-1)) ? Frecv[BUFIDX(4,k-nzl+l,i,j,3,nxl,nyl)] : et[ IDX(i,j,k+l,nxl,nyl,nzl)];
    if (nchem>0) {
      for (int l=0; l<3; l++) {
        if (k<(3-l)) {
          for (int v=0; v<nchem; v++)  w1d[l][5+v] = Brecv[BUFIDX(5+v,k+l,i,j,3,nxl,nyl)];
        } else {
          realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,k-3+l,nxl,nyl,nzl));
          if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_z_bdry)", 0)) throw std::exception();
          for (int v=0; v<nchem; v++)  w1d[l][5+v] = chem[v];
        }
      }
      for (int l=0; l<3; l++) {
        if (k>(nzl-l-1)) {
          for (int v=0; v<nchem; v++)  w1d[l+3][5+v] = Frecv[BUFIDX(5+v,k-nzl+l,i,j,3,nxl,nyl)];
        } else {
          realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+IDX(i,j,k+l,nxl,nyl,nzl));
          if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (pack1D_z_bdry)", 0)) throw std::exception();
          for (int v=0; v<nchem; v++)  w1d[l+3][5+v] = chem[v];
        }
      }
    }
  }

  // Equation of state -- compute and return pressure,
  //    p = (gamma-1)*(e - rho/2*(vx^2+vy^2+vz^2)), or equivalently
  //    p = (gamma-1)*(e - (mx^2+my^2+mz^2))/(2*rho)
  inline realtype eos(const realtype& rho, const realtype& mx,
                      const realtype& my, const realtype& mz,
                      const realtype& et) const 
  {
    return((gamma-ONE)*(et - (mx*mx+my*my+mz*mz)*HALF/rho));
  }

  // Equation of state inverse -- compute and return energy,
  //    e_t = p/(gamma-1) + rho/2*(v_x^2 + v_y^2 + v_z^2), or equivalently
  //    e_t = p/(gamma-1) + (m_x^2 + m_y^2 + m_z^2)/(2*rho)
  inline realtype eos_inv(const realtype& rho, const realtype& mx,
                          const realtype& my, const realtype& mz,
                          const realtype& pr) const 
  {
    return(pr/(gamma-ONE) + (mx*mx+my*my+mz*mz)*HALF/rho);
  }

  // Check for legal state: returns 0 if density, energy and pressure
  // are all positive;  otherwise the return value encodes all failed
  // variables:  dfail + efail + pfail, with dfail=0/1, efail=0/2, pfail=0/4,
  // i.e., a return of 5 indicates that density and pressure were
  // non-positive, but energy was fine
  inline int legal_state(const realtype& rho, const realtype& mx,
                         const realtype& my, const realtype& mz,
                         const realtype& et) const 
  {
    int dfail, efail, pfail;
    dfail = (rho > ZERO) ? 0 : 1;
    efail = (et > ZERO) ? 0 : 2;
    pfail = (eos(rho, mx, my, mz, et) > ZERO) ? 0 : 4;
    return(dfail+efail+pfail);
  }
  
};   // end UserData;




// Additional utility routines

//    Load inputs from file
int load_inputs(int myid, int argc, char* argv[], UserData& udata, ARKodeParameters& opts);

//    Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const UserData& udata);

//    Forcing terms
int external_forces(const realtype& t, N_Vector G, const UserData& udata);

//    Output solution diagnostics
int output_diagnostics(const realtype& t, const N_Vector w, const UserData& udata);

//    Optional conservation checks
int check_conservation(const realtype& t, const N_Vector w, const UserData& udata);

//    Print solution statistics
int print_stats(const realtype& t, const N_Vector w, const int& firstlast, 
                void *arkode_mem, const UserData& udata);

//    Output information on domain and this subdomain
int output_subdomain_information(const UserData& udata, const realtype& dTout);

//    Output current solution
int output_solution(const N_Vector w, const int& newappend,
                    const UserData& udata);

//    WENO Div(flux(u)) function
void face_flux(realtype (&w1d)[6][NVAR], const int& idir,
               realtype* f_face, const UserData& udata);

//    Routine to compute the Euler ODE RHS function f(t,y).
int fEuler(realtype t, N_Vector w, N_Vector wdot, void* user_data);

//    CFL stable time step calculation routine
int stability(N_Vector w, realtype t, realtype* dt_stab, void* user_data);
  
#endif
//---- end of file ----
