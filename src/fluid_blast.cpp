/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Test problem in which a blast wave proceeds across a "clumpy"
 density field.  The initial density field is defined to be

    rho(X) = rho0*(1 + \sum_i s_i*exp(-2*(||X-X_i||/r_i)^2)),

 where s_i, r_i and X_i are clump-dependent.  We place these
 throughout the domain by randomly choosing CLUMPS_PER_PROC*nprocs
 overall clumps in the simulation box; while this is based on a
 uniform distribution, no process is guaranteed to have
 CLUMPS_PER_PROC clumps centered within its domain.  We randomly
 choose the clump "radius" r_i to equal a uniformly-distributed
 random number in the interval
 [dx*MIN_CLUMP_RADIUS, dx*MAX_CLUMP_RADIUS].  Finally, we randomly
 choose the clump "strength" s_i to be a uniformly-distributed
 random number in the interval [0, MAX_CLUMP_STRENGTH].  The
 parameters CLUMPS_PER_PROC, MIN_CLUMP_RADIUS, MAX_CLUMP_RADIUS
 and MAX_CLUMP_STRENGTH are #defined below.

 This test is set up to mirror primordial_blast.cpp, except that
 here all chemistry is disabled.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <random>

// basic problem definitions
#define  CLUMPS_PER_PROC     10             // on average
#define  MIN_CLUMP_RADIUS    RCONST(2.0)    // in number of cells
#define  MAX_CLUMP_RADIUS    RCONST(4.0)    // in number of cells
#define  MAX_CLUMP_STRENGTH  RCONST(100.0)  // mult. density factor


// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const EulerData& udata)
{

  // access data fields
  long int i, j, k, idx;
  int retval;
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (initial_conditions)", 0)) return -1;

  // output test problem information
  if (udata.myid == 0)
    cout << "\nPrimordial blast test problem\n\n";

  // ensure that this is compiled without chemical species
  if (udata.nchem > 0) {
    cerr << "\nThis test should not include chemistry (compiled with "
         << udata.nchem << " chemical fields), exiting\n\n";
    return -1;
  }

  // return error if input parameters are inappropriate
  if ( (udata.xlbc != BC_REFLECTING) || (udata.xrbc != BC_REFLECTING) ||
       (udata.ylbc != BC_REFLECTING) || (udata.yrbc != BC_REFLECTING) ||
       (udata.zlbc != BC_REFLECTING) || (udata.zrbc != BC_REFLECTING) ) {
    cerr << "\nInappropriate boundary conditions (should be reflecting), exiting\n\n";
    return -1;
  }

  // root process determines locations, radii and strength of density clumps
  long int nclumps = CLUMPS_PER_PROC*udata.nprocs;
  double clump_data[nclumps*5];
  if (udata.myid == 0) {

    // initialize mersenne twister with seed equal to the number of MPI ranks (for reproducibility)
    std::mt19937_64 gen(udata.nprocs);
    std::uniform_real_distribution<> cx_d(udata.xl, udata.xr);
    std::uniform_real_distribution<> cy_d(udata.yl, udata.yr);
    std::uniform_real_distribution<> cz_d(udata.zl, udata.zr);
    std::uniform_real_distribution<> cr_d(MIN_CLUMP_RADIUS,MAX_CLUMP_RADIUS);
    std::uniform_real_distribution<> cs_d(ZERO, MAX_CLUMP_STRENGTH);

    // fill clump information
    for (i=0; i<nclumps; i++) {

      // global (x,y,z) coordinates for this clump center
      clump_data[5*i+0] = cx_d(gen);
      clump_data[5*i+1] = cy_d(gen);
      clump_data[5*i+2] = cz_d(gen);

      // radius of clump
      clump_data[5*i+3] = cr_d(gen);

      // strength of clump
      clump_data[5*i+4] = cs_d(gen);

    }

  }

  // root process broadcasts clump information
  retval = MPI_Bcast(clump_data, nclumps*5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (initial_conditions)", 3)) return -1;


  // output clump information
  if (udata.myid == 0) {
    cout << "\nInitializing problem with " << nclumps << " clumps:\n";
    for (i=0; i<nclumps; i++)
      cout << "   clump " << i << ", center = (" << clump_data[5*i+0] << ","
           << clump_data[5*i+1] << "," << clump_data[5*i+2] << "),  \tradius = "
           << clump_data[5*i+3] << " cells,  \tstrength = " << clump_data[5*i+4] << std::endl;
  }


  // initial condition values -- essentially-neutral primordial gas
  // realtype Tmean = 2000.0;  // mean temperature in K
  realtype Tmean = 10.0;  // mean temperature in K
  // realtype tiny = 1e-20;
  realtype tiny = 1e-40;
  realtype mH = 1.67e-24;
  realtype Hfrac = 0.76;
  realtype HI_weight = 1.00794 * mH;
  realtype HII_weight = 1.00794 * mH;
  realtype HM_weight = 1.00794 * mH;
  realtype HeI_weight = 4.002602 * mH;
  realtype HeII_weight = 4.002602 * mH;
  realtype HeIII_weight = 4.002602 * mH;
  realtype H2I_weight = 2*HI_weight;
  realtype H2II_weight = 2*HI_weight;
  realtype kboltz = 1.3806488e-16;
  realtype H2I, H2II, HI, HII, HM, HeI, HeII, HeIII, de, T, ge;
  realtype nH2I, nH2II, nHI, nHII, nHM, nHeI, nHeII, nHeIII, ndens;
  realtype density0 = 1e2 * mH;   // in g/cm^{-3}
  realtype density, xloc, yloc, zloc, cx, cy, cz, cr, cs, xdist, ydist, zdist;

  // iterate over subdomain, setting initial conditions
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {

        // determine cell center
        xloc = (udata.is+i+HALF)*udata.dx + udata.xl;
        yloc = (udata.js+j+HALF)*udata.dy + udata.yl;
        zloc = (udata.ks+k+HALF)*udata.dz + udata.zl;

        // determine density in this cell (via loop over clumps)
        density = ONE;
        for (idx=0; idx<nclumps; idx++) {
          cx = clump_data[5*idx+0];
          cy = clump_data[5*idx+1];
          cz = clump_data[5*idx+2];
          cr = clump_data[5*idx+3]*udata.dx;
          cs = clump_data[5*idx+4];
          // xdist = min( abs(xloc-cx), min( abs(xloc-cx+udata.xr), abs(xloc-cx-udata.xr) ) );
          // ydist = min( abs(yloc-cy), min( abs(yloc-cx+udata.yr), abs(xloc-cx-udata.xr) ) );
          // zdist = min( abs(zloc-cz), min( abs(zloc-cx+udata.zr), abs(xloc-cx-udata.xr) ) );
          xdist = abs(xloc-cx);
          ydist = abs(yloc-cy);
          zdist = abs(zloc-cz);
          density += cs*exp(-2.0*(xdist*xdist + ydist*ydist + zdist*zdist)/cr/cr);
        }
        density *= density0;

        // set mass densities into local variables
        H2I = 1.e-3*density;
        H2II = tiny*density;
        HII = 1.e2*tiny*density;
        HM = tiny*density;
        HeII = tiny*density;
        HeIII = tiny*density;
        HeI = (ONE-Hfrac)*density - HeII - HeIII;
        HI = density - (H2I+H2II+HII+HM+HeI+HeII+HeIII);

        // compute derived number densities
        nH2I   = H2I   / H2I_weight;
        nH2II  = H2II  / H2II_weight;
        nHII   = HII   / HII_weight;
        nHM    = HM    / HM_weight;
        nHeII  = HeII  / HeII_weight;
        nHeIII = HeIII / HeIII_weight;
        nHeI   = HeI   / HeI_weight;
        nHI    = HI    / HI_weight;
        ndens  = nH2I + nH2II + nHII + nHM + nHeII + nHeIII + nHeI + nHI;

        // set location-dependent temperature, and convert to gas energy
        T = Tmean;
        ge = (kboltz * T * ndens) / (density * (udata.gamma - ONE));

        // set hydrodynamic initial conditions in dimensionless units
        idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
        rho[idx] = density/udata.DensityUnits;
        mx[idx]  = ZERO/udata.MomentumUnits;
        my[idx]  = ZERO/udata.MomentumUnits;
        mz[idx]  = ZERO/udata.MomentumUnits;
        et[idx]  = (ge + 0.5/rho[idx]*(mx[idx]*mx[idx] + my[idx]*my[idx] + mz[idx]*mz[idx]))
                 / udata.EnergyUnits;

      }

  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  // initialize external forces to zero
  N_VConst(ZERO, G);
  return 0;
}

// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const EulerData& udata)
{
  // return with success
  return(0);
}

//---- end of file ----