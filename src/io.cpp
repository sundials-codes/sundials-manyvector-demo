/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Implementation file for input/output utility routines.   
 Input routines read problem and solver input parameters
 from specified files.  For solver parameters, this calls
 associated "set" routines to specify options to ARKode.
 Output routines compute/output shared diagnostics information,
 or write solution data to disk.
 ---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>
#include <string.h>
#include "gopt.h"


#define MAX_LINE_LENGTH 512


// Load problem-defining parameters from file: root process
// reads parameters and broadcasts results to remaining
// processes
int load_inputs(int myid, int argc, char* argv[],
                UserData& udata, ARKodeParameters& opts)
{
  int retval;
  double dbuff[23];
  long int ibuff[19];

  // root process handles command-line and file-based solver parameters, and packs send buffers
  if (myid == 0) {

    // use 'gopt' to handle parsing command-line; first define all available options
    const int nopt = 44;
    struct option options[nopt+1];
    enum iarg { ifname, ihelp, ixl, ixr, iyl, iyr, izl, izr, it0, 
                itf, igam, inx, iny, inz, ixlb, ixrb, iylb, 
                iyrb, izlb, izrb, icfl, inout, ishow,
                iord, idord, ibt, iadmth, imnef, imhnil, imaxst,
                isfty, ibias, igrow, ipq, ik1, ik2, ik3, iemx1,
                iemaf, ih0, ihmin, ihmax, irtol, iatol};
    for (int i=0; i<nopt; i++) {
      options[i].short_name = '0';
      options[i].flags = GOPT_ARGUMENT_REQUIRED;
    }
    options[nopt].flags = GOPT_LAST;
    options[ifname].short_name = 'f';
    options[ifname].long_name = "infile";
    options[ihelp].short_name = 'h';
    options[ihelp].long_name = "help";
    options[ihelp].flags = GOPT_ARGUMENT_FORBIDDEN;
    options[ixl].long_name = "xl";
    options[ixr].long_name = "xr";
    options[iyl].long_name = "yl";
    options[iyr].long_name = "yr";
    options[izl].long_name = "zl";
    options[izr].long_name = "zr";
    options[it0].long_name = "t0";
    options[itf].long_name = "tf";
    options[igam].long_name = "gamma";
    options[inx].long_name = "nx";
    options[iny].long_name = "ny";
    options[inz].long_name = "nz";
    options[ixlb].long_name = "xlbc";
    options[ixrb].long_name = "xrbc";
    options[iylb].long_name = "ylbc";
    options[iyrb].long_name = "yrbc";
    options[izlb].long_name = "zlbc";
    options[izrb].long_name = "zrbc";
    options[icfl].long_name = "cfl";
    options[inout].long_name = "nout";
    options[ishow].long_name = "showstats";
    options[iord].long_name = "order";
    options[idord].long_name = "dense_order";
    options[ibt].long_name = "btable";
    options[iadmth].long_name = "adapt_method";
    options[imnef].long_name = "maxnef";
    options[imhnil].long_name = "mxhnil";
    options[imaxst].long_name = "mxsteps";
    options[isfty].long_name = "safety";
    options[ibias].long_name = "bias";
    options[igrow].long_name = "growth";
    options[ipq].long_name = "pq";
    options[ik1].long_name = "k1";
    options[ik2].long_name = "k2";
    options[ik3].long_name = "k3";
    options[iemx1].long_name = "etamx1";
    options[iemaf].long_name = "etamxf";
    options[ih0].long_name = "h0";
    options[ihmin].long_name = "hmin";
    options[ihmax].long_name = "hmax";
    options[irtol].long_name = "rtol";
    options[iatol].long_name = "atol";
    argc = gopt(argv, options);
    //gopt_errors(argv[0], options);

    // handle help request
    if (options[ihelp].count) {
      cout << "\nEuler3D SUNDIALS ManyVector+Multirate demonstration code\n"
           << "\nUsage: " << argv[0] << " [options]\n"
           << "   -h or --help prints this message and exits the program\n"
           << "\nAvailable problem specification options (and the default if not provided):\n"
           << "   --xl=<float>         (" << udata.xl << ")\n"
           << "   --xr=<float>         (" << udata.xr << ")\n"
           << "   --yl=<float>         (" << udata.yl << ")\n"
           << "   --yr=<float>         (" << udata.yr << ")\n"
           << "   --zl=<float>         (" << udata.zl << ")\n"
           << "   --zr=<float>         (" << udata.zr << ")\n"
           << "   --t0=<float>         (" << udata.t0 << ")\n"
           << "   --tf=<float>         (" << udata.tf << ")\n"
           << "   --gamma=<float>      (" << udata.gamma << ")\n"
           << "   --nx=<int>           (" << udata.nx << ")\n"
           << "   --ny=<int>           (" << udata.ny << ")\n"
           << "   --nz=<int>           (" << udata.nz << ")\n"
           << "   --xlbc=<int>         (" << udata.xlbc << ")\n"
           << "   --xrbc=<int>         (" << udata.xrbc << ")\n"
           << "   --ylbc=<int>         (" << udata.ylbc << ")\n"
           << "   --yrbc=<int>         (" << udata.yrbc << ")\n"
           << "   --zlbc=<int>         (" << udata.zlbc << ")\n"
           << "   --zrbc=<int>         (" << udata.zrbc << ")\n"
           << "\nAvailable run options (and the default if not provided):\n"
           << "   --nout=<int>         (" << udata.nout << ")\n"
           << "   --showstats          (disabled)\n"
           << "\nAvailable time-stepping options (and the default if not provided):\n"
           << "   --cfl=<float>        (" << udata.cfl << ")\n"
           << "   --order=<int>        (" << opts.order << ")\n"
           << "   --dense_order=<int>  (" << opts.dense_order << ")\n"
           << "   --btable=<int>       (" << opts.btable << ")\n"
           << "   --adapt_method=<int> (" << opts.adapt_method << ")\n"
           << "   --maxnef=<int>       (" << opts.maxnef << ")\n"
           << "   --mxhnil=<int>       (" << opts.mxhnil << ")\n"
           << "   --mxsteps=<int>      (" << opts.mxsteps << ")\n"
           << "   --safety=<float>     (" << opts.safety << ")\n"
           << "   --bias=<float>       (" << opts.bias << ")\n"
           << "   --growth=<float>     (" << opts.growth << ")\n"
           << "   --pq=<int>           (" << opts.pq << ")\n"
           << "   --k1=<float>         (" << opts.k1 << ")\n"
           << "   --k2=<float>         (" << opts.k2 << ")\n"
           << "   --k3=<float>         (" << opts.k3 << ")\n"
           << "   --etamx1=<float>     (" << opts.etamx1 << ")\n"
           << "   --etamxf=<float>     (" << opts.etamxf << ")\n"
           << "   --h0=<float>         (" << opts.h0 << ")\n"
           << "   --hmin=<float>       (" << opts.hmin << ")\n"
           << "   --hmax=<float>       (" << opts.hmax << ")\n"
           << "   --rtol=<float>       (" << opts.rtol << ")\n"
           << "   --atol=<float>       (" << opts.atol << ")\n"
           << "\nAlternately, all of these options may be specified in a single\n"
           << "input file (with command-line arguments taking precedence if an\n"
           << "option is multiply-defined) via:"
           << "   -f <fname> or --infile=<fname>\n\n\n";
      return(1);
    }
    
    // if an input file was specified, read that here
    if (options[ifname].count) {
      char line[MAX_LINE_LENGTH];
      FILE *FID=NULL;
      FID = fopen(options[ifname].argument,"r");
      if (check_flag((void *) FID, "fopen (load_inputs)", 0)) return(-1);
      while (fgets(line, MAX_LINE_LENGTH, FID) != NULL) {

        /* initialize return flag for line */
        retval = 0;

        /* read parameters */
        retval += sscanf(line,"xl = %lf", &udata.xl);
        retval += sscanf(line,"xr = %lf", &udata.xr);
        retval += sscanf(line,"yl = %lf", &udata.yl);
        retval += sscanf(line,"yr = %lf", &udata.yr);
        retval += sscanf(line,"zl = %lf", &udata.zl);
        retval += sscanf(line,"zr = %lf", &udata.zr);
        retval += sscanf(line,"t0 = %lf", &udata.t0);
        retval += sscanf(line,"tf = %lf", &udata.tf);
        retval += sscanf(line,"gamma = %lf", &udata.gamma);
        retval += sscanf(line,"nx = %li", &udata.nx);
        retval += sscanf(line,"ny = %li", &udata.ny);
        retval += sscanf(line,"nz = %li", &udata.nz);
        retval += sscanf(line,"xlbc = %i", &udata.xlbc);
        retval += sscanf(line,"xrbc = %i", &udata.xrbc);
        retval += sscanf(line,"ylbc = %i", &udata.ylbc);
        retval += sscanf(line,"yrbc = %i", &udata.yrbc);
        retval += sscanf(line,"zlbc = %i", &udata.zlbc);
        retval += sscanf(line,"zrbc = %i", &udata.zrbc);
        retval += sscanf(line,"cfl = %lf", &udata.cfl);
        retval += sscanf(line,"nout = %i", &udata.nout);
        retval += sscanf(line,"showstats = %i", &udata.showstats);
        retval += sscanf(line,"order = %i", &opts.order);
        retval += sscanf(line,"dense_order = %i", &opts.dense_order);
        retval += sscanf(line,"btable = %i",  &opts.btable);
        retval += sscanf(line,"adapt_method = %i", &opts.adapt_method);
        retval += sscanf(line,"maxnef = %i", &opts.maxnef);
        retval += sscanf(line,"mxhnil = %i", &opts.mxhnil);
        retval += sscanf(line,"mxsteps = %i", &opts.mxsteps);
        retval += sscanf(line,"safety = %lf", &opts.safety);
        retval += sscanf(line,"bias = %lf", &opts.bias);
        retval += sscanf(line,"growth = %lf", &opts.growth);
        retval += sscanf(line,"pq = %i", &opts.pq);
        retval += sscanf(line,"k1 = %lf", &opts.k1);
        retval += sscanf(line,"k2 = %lf", &opts.k2);
        retval += sscanf(line,"k3 = %lf", &opts.k3);
        retval += sscanf(line,"etamx1 = %lf", &opts.etamx1);
        retval += sscanf(line,"etamxf = %lf", &opts.etamxf);
        retval += sscanf(line,"h0 = %lf", &opts.h0);
        retval += sscanf(line,"hmin = %lf", &opts.hmin);
        retval += sscanf(line,"hmax = %lf", &opts.hmax);
        retval += sscanf(line,"rtol = %lf", &opts.rtol);
        retval += sscanf(line,"atol = %lf", &opts.atol);

        /* if unable to read the line (and it looks suspicious) issue a warning */
        if (retval == 0 && strstr(line, "=") != NULL && line[0] != '#')
          fprintf(stderr, "load_inputs Warning: parameter line was not interpreted:\n%s", line);
      }
      fclose(FID);
      
    }

    // replace any current option with a value specified on the command line
    if (options[ixl].count)    udata.xl          = atof(options[ixl].argument);
    if (options[ixr].count)    udata.xr          = atof(options[ixr].argument);
    if (options[iyl].count)    udata.yl          = atof(options[iyl].argument);
    if (options[iyr].count)    udata.yr          = atof(options[iyr].argument);
    if (options[izl].count)    udata.zl          = atof(options[izl].argument);
    if (options[izr].count)    udata.zr          = atof(options[izr].argument);
    if (options[it0].count)    udata.t0          = atof(options[it0].argument);
    if (options[itf].count)    udata.tf          = atof(options[itf].argument);
    if (options[igam].count)   udata.gamma       = atof(options[igam].argument);
    if (options[inx].count)    udata.nx          = atoi(options[inx].argument);
    if (options[iny].count)    udata.ny          = atoi(options[iny].argument);
    if (options[inz].count)    udata.nz          = atoi(options[inz].argument);
    if (options[ixlb].count)   udata.xlbc        = atoi(options[ixlb].argument);
    if (options[ixrb].count)   udata.xrbc        = atoi(options[ixrb].argument);
    if (options[iylb].count)   udata.ylbc        = atoi(options[iylb].argument);
    if (options[iyrb].count)   udata.yrbc        = atoi(options[iyrb].argument);
    if (options[izlb].count)   udata.zlbc        = atoi(options[izlb].argument);
    if (options[izrb].count)   udata.zrbc        = atoi(options[izrb].argument);
    if (options[icfl].count)   udata.cfl         = atof(options[icfl].argument);
    if (options[inout].count)  udata.nout        = atoi(options[inout].argument);
    if (options[ishow].count)  udata.showstats = 1;
    if (options[iord].count)   opts.order        = atoi(options[iord].argument);
    if (options[idord].count)  opts.dense_order  = atoi(options[idord].argument);
    if (options[ibt].count)    opts.btable       = atoi(options[ibt].argument);
    if (options[iadmth].count) opts.adapt_method = atoi(options[iadmth].argument);
    if (options[imnef].count)  opts.maxnef       = atoi(options[imnef].argument);
    if (options[imhnil].count) opts.mxhnil       = atoi(options[imhnil].argument);
    if (options[imaxst].count) opts.mxsteps      = atoi(options[imaxst].argument);
    if (options[isfty].count)  opts.safety       = atof(options[isfty].argument);
    if (options[ibias].count)  opts.bias         = atof(options[ibias].argument);
    if (options[igrow].count)  opts.growth       = atof(options[igrow].argument);
    if (options[ipq].count)    opts.pq           = atoi(options[ipq].argument);
    if (options[ik1].count)    opts.k1           = atof(options[ik1].argument);
    if (options[ik2].count)    opts.k2           = atof(options[ik2].argument);
    if (options[ik3].count)    opts.k3           = atof(options[ik3].argument);
    if (options[iemx1].count)  opts.etamx1       = atof(options[iemx1].argument);
    if (options[iemaf].count)  opts.etamxf       = atof(options[iemaf].argument);
    if (options[ih0].count)    opts.h0           = atof(options[ih0].argument);
    if (options[ihmin].count)  opts.hmin         = atof(options[ihmin].argument);
    if (options[ihmax].count)  opts.hmax         = atof(options[ihmax].argument);
    if (options[irtol].count)  opts.rtol         = atof(options[irtol].argument);
    if (options[iatol].count)  opts.atol         = atof(options[iatol].argument);

    // pack buffers with final parameter values
    ibuff[0]  = udata.nx;
    ibuff[1]  = udata.ny;
    ibuff[2]  = udata.nz;
    ibuff[3]  = udata.xlbc;
    ibuff[4]  = udata.xrbc;
    ibuff[5]  = udata.ylbc;
    ibuff[6]  = udata.yrbc;
    ibuff[7]  = udata.zlbc;
    ibuff[8]  = udata.zrbc;
    ibuff[9]  = udata.nout;
    ibuff[10] = udata.showstats;
    ibuff[11] = opts.order;
    ibuff[12] = opts.dense_order;
    ibuff[13] = opts.btable;
    ibuff[14] = opts.adapt_method;
    ibuff[15] = opts.maxnef;
    ibuff[16] = opts.mxhnil;
    ibuff[17] = opts.mxsteps;
    ibuff[18] = opts.pq;

    dbuff[0]  = udata.xl;
    dbuff[1]  = udata.xr;
    dbuff[2]  = udata.yl;
    dbuff[3]  = udata.yr;
    dbuff[4]  = udata.zl;
    dbuff[5]  = udata.zr;
    dbuff[6]  = udata.t0;
    dbuff[7]  = udata.tf;
    dbuff[8]  = udata.gamma;
    dbuff[9]  = udata.cfl;
    dbuff[10] = opts.safety;
    dbuff[11] = opts.bias;
    dbuff[12] = opts.growth;
    dbuff[13] = opts.k1;
    dbuff[14] = opts.k2;
    dbuff[15] = opts.k3;
    dbuff[16] = opts.etamx1;
    dbuff[17] = opts.etamxf;
    dbuff[18] = opts.h0;
    dbuff[19] = opts.hmin;
    dbuff[20] = opts.hmax;
    dbuff[21] = opts.rtol;
    dbuff[22] = opts.atol;
  }

  // perform broadcast and unpack results
  retval = MPI_Bcast(dbuff, 23, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (load_inputs)", 3)) return(-1);
  retval = MPI_Bcast(ibuff, 19, MPI_LONG, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (load_inputs)", 3)) return(-1);

  // unpack buffers
  udata.nx = ibuff[0];
  udata.ny = ibuff[1];
  udata.nz = ibuff[2];
  udata.xlbc = ibuff[3];
  udata.xrbc = ibuff[4];
  udata.ylbc = ibuff[5];
  udata.yrbc = ibuff[6];
  udata.zlbc = ibuff[7];
  udata.zrbc = ibuff[8];
  udata.nout = ibuff[9];
  udata.showstats = ibuff[10];
  opts.order = ibuff[11];
  opts.dense_order = ibuff[12];
  opts.btable = ibuff[13];
  opts.adapt_method = ibuff[14];
  opts.maxnef = ibuff[15];
  opts.mxhnil = ibuff[16];
  opts.mxsteps = ibuff[17];
  opts.pq = ibuff[18];
  
  udata.xl = dbuff[0];
  udata.xr = dbuff[1];
  udata.yl = dbuff[2];
  udata.yr = dbuff[3];
  udata.zl = dbuff[4];
  udata.zr = dbuff[5];
  udata.t0 = dbuff[6];
  udata.tf = dbuff[7];
  udata.gamma = dbuff[8];
  udata.cfl = dbuff[9];
  opts.safety = dbuff[10];
  opts.bias = dbuff[11];
  opts.growth = dbuff[12];
  opts.k1 = dbuff[13];
  opts.k2 = dbuff[14];
  opts.k3 = dbuff[15];
  opts.etamx1 = dbuff[16];
  opts.etamxf = dbuff[17];
  opts.h0 = dbuff[18];
  opts.hmin = dbuff[19];
  opts.hmax = dbuff[20];
  opts.rtol = dbuff[21];
  opts.atol = dbuff[22];

  // return with success
  return(0);
}


// Computes the total of each conserved quantity;
// the root task then outputs these values to screen
int check_conservation(const realtype& t, const N_Vector w, const UserData& udata)
{
  realtype sumvals[] = {ZERO, ZERO};
  realtype totvals[] = {ZERO, ZERO};
  static realtype totsave[] = {-ONE, -ONE};
  bool outproc = (udata.myid == 0);
  long int i, j, k, idx;
  int retval;
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (check_conservation)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (check_conservation)", 0)) return -1;
  for (k=0; k<udata.nzl; k++)
    for (j=0; j<udata.nyl; j++)
      for (i=0; i<udata.nxl; i++) {
        idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
        sumvals[0] += rho[idx];
        sumvals[1] += et[idx];
      }
  sumvals[0] *= udata.dx*udata.dy*udata.dz;
  sumvals[1] *= udata.dx*udata.dy*udata.dz;
  retval = MPI_Reduce(sumvals, totvals, 2, MPI_SUNREALTYPE, MPI_SUM, 0, udata.comm);
  if (check_flag(&retval, "MPI_Reduce (check_conservation)", 3)) MPI_Abort(udata.comm, 1);
  if (!outproc)  return(0);
  if (totsave[0] == -ONE) {  // first time through; save/output the values
    printf("   Total mass   = %21.16e\n", totvals[0]);
    printf("   Total energy = %21.16e\n", totvals[1]);
    totsave[0] = totvals[0];
    totsave[1] = totvals[1];
  } else {
    printf("   Mass conservation relative change   = %7.2e\n",
           abs(totvals[0]-totsave[0])/totsave[0]);
    printf("   Energy conservation relative change = %7.2e\n",
           abs(totvals[1]-totsave[1])/totsave[1]);
  }
  return(0);
}


// Utility routine to print solution statistics
//    firstlast = 0 indicates the first output
//    firstlast = 1 indicates a normal output
//    firstlast = 2 indicates the lastoutput
int print_stats(const realtype& t, const N_Vector w,
                const int& firstlast, const UserData& udata)
{
  realtype rmsvals[NVAR], totrms[NVAR];
  bool outproc = (udata.myid == 0);
  long int v, i, j, k, idx;
  int retval;
  realtype *rho = N_VGetSubvectorArrayPointer_MPIManyVector(w,0);
  if (check_flag((void *) rho, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *mx = N_VGetSubvectorArrayPointer_MPIManyVector(w,1);
  if (check_flag((void *) mx, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *my = N_VGetSubvectorArrayPointer_MPIManyVector(w,2);
  if (check_flag((void *) my, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *mz = N_VGetSubvectorArrayPointer_MPIManyVector(w,3);
  if (check_flag((void *) mz, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  realtype *et = N_VGetSubvectorArrayPointer_MPIManyVector(w,4);
  if (check_flag((void *) et, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
  if (firstlast < 2) {
    for (v=0; v<NVAR; v++)  rmsvals[v] = ZERO;
    for (k=0; k<udata.nzl; k++)
      for (j=0; j<udata.nyl; j++)
        for (i=0; i<udata.nxl; i++) {
          idx = IDX(i,j,k,udata.nxl,udata.nyl,udata.nzl);
          rmsvals[0] += pow(rho[idx], 2);
          rmsvals[1] += pow( mx[idx], 2);
          rmsvals[2] += pow( my[idx], 2);
          rmsvals[3] += pow( mz[idx], 2);
          rmsvals[4] += pow( et[idx], 2);
          if (udata.nchem > 0) {
            realtype *chem = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+idx);
            if (check_flag((void *) chem, "N_VGetSubvectorArrayPointer (print_stats)", 0)) return -1;
            for (v=0; v<udata.nchem; v++)
              rmsvals[5+v] += SUNRpowerI( chem[v], 2);
          }
        }
    retval = MPI_Reduce(rmsvals, totrms, NVAR, MPI_SUNREALTYPE, MPI_SUM, 0, udata.comm);
    if (check_flag(&retval, "MPI_Reduce (print_stats)", 3)) MPI_Abort(udata.comm, 1);
    for (v=0; v<NVAR; v++)  totrms[v] = SUNRsqrt(totrms[v]/udata.nx/udata.ny/udata.nz);
  }
  if (!outproc)  return(0);
  if (firstlast == 0) {
    cout << "\n        t     ||rho||_rms  ||mx||_rms  ||my||_rms  ||mz||_rms  ||et||_rms";
    for (v=0; v<udata.nchem; v++)  cout << "  ||c" << v << "||_rms";
    cout << endl;
  }
  if (firstlast != 1) {
    cout << "   -----------------------------------------------------------------------";
    for (v=0; v<udata.nchem; v++)  cout << "-----------";
    cout << endl;
  }
  if (firstlast<2) {
    printf("  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f", t,
           totrms[0], totrms[1], totrms[2], totrms[3], totrms[4]);
    for (v=0; v<udata.nchem; v++)  printf("  %10.6f", totrms[5+v]);
    printf("\n");
  }
  return(0);
}


// Utility routine to output information on this subdomain
int output_subdomain_information(const UserData& udata, const realtype& dTout)
{
  char outname[100];
  FILE *UFID = NULL;
  sprintf(outname, "output-subdomain.%07i.txt", udata.myid);
  UFID = fopen(outname,"w");
  if (check_flag((void*) UFID, "fopen (output_subdomain_information)", 0)) return(1);
  fprintf(UFID, "%li  %li  %li  %li  %li  %li  %li  %li  %li  %i  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf\n",
	  udata.nx, udata.ny, udata.nz, udata.is, udata.ie, udata.js, udata.je, 
          udata.ks, udata.ke, udata.nchem, udata.xl, udata.xr, udata.yl, 
          udata.yr, udata.zl, udata.zr, udata.t0, udata.tf, dTout);
  fclose(UFID);
  return(0);
}


// Utility routine to output the current solution
//    newappend == 1 indicates create a new file
//    newappend == 0 indicates append to existing file
int output_solution(const N_Vector w, const int& newappend, const UserData& udata)
{
  // reusable variables
  char outtype[2];
  char outname[NVAR][100];
  FILE *FID[NVAR];
  realtype *W;
  long int i, v;
  long int N = (udata.nzl)*(udata.nyl)*(udata.nxl);

  // Set string for output type
  if (newappend == 1) {
    sprintf(outtype, "w");
  } else {
    sprintf(outtype, "a");
  }

  // Set strings for output names
  sprintf(outname[0], "output-rho.%07i.txt", udata.myid);  // density
  sprintf(outname[1], "output-mx.%07i.txt",  udata.myid);  // x-momentum
  sprintf(outname[2], "output-my.%07i.txt",  udata.myid);  // y-momentum
  sprintf(outname[3], "output-mz.%07i.txt",  udata.myid);  // z-momentum
  sprintf(outname[4], "output-et.%07i.txt",  udata.myid);  // total energy
  //   tracers -- set index width based on total number (assumes at most 10001 tracer species)
  if (udata.nchem < 11) {
    for (v=0; v<udata.nchem; v++)
      sprintf(outname[5+v], "output-c%01i.%07i.txt", (int) v, udata.myid);
  } else if (udata.nchem < 101) {
    for (v=0; v<udata.nchem; v++)
      sprintf(outname[5+v], "output-c%02i.%07i.txt", (int) v, udata.myid);
  } else if (udata.nchem < 1001) {
    for (v=0; v<udata.nchem; v++)
      sprintf(outname[5+v], "output-c%03i.%07i.txt", (int) v, udata.myid);
  } else if (udata.nchem < 10001) {
    for (v=0; v<udata.nchem; v++)
      sprintf(outname[5+v], "output-c%04i.%07i.txt", (int) v, udata.myid);
  } else {
    cerr << "output_solution error: cannot handle over 10000 tracer species\n";
    return(1);
  }
  // Output fluid fields to disk
  for (v=0; v<5; v++) {
    FID[v] = fopen(outname[v],outtype);                  // file ptr
    W = N_VGetSubvectorArrayPointer_MPIManyVector(w,v);  // data array
    if (check_flag((void *) W, "N_VGetSubvectorArrayPointer (output_solution)", 0)) return -1;
    for (i=0; i<N; i++) fprintf(FID[v]," %.16e", W[i]);  // output
    fprintf(FID[v],"\n");                                // newline
    fclose(FID[v]);                                      // close file
  }

  // Output tracer fields to disk
  if (udata.nchem > 0) {
    for (v=0; v<udata.nchem; v++)                        // open file ptrs
      FID[v] = fopen(outname[5+v],outtype);
    for (i=0; i<N; i++) {                                // loop over subdomain
      W = NULL;
      W = N_VGetSubvectorArrayPointer_MPIManyVector(w,5+i);
      if (check_flag((void *) W, "N_VGetSubvectorArrayPointer (output_solution)", 0)) return -1;
      for (v=0; v<udata.nchem; v++)                      // output tracers at this location
        fprintf(FID[v]," %.16e", W[v]);
    }
    for (v=0; v<udata.nchem; v++) {                      // add newlines and close files
      fprintf(FID[v],"\n");
      fclose(FID[v]);
    }
  }
  
  // return with success
  return(0);
}


//---- end of file ----
