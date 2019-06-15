/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Utility routines to read problem and solver input parameters
 from specified files.  For solver parameters, this calls
 associated "set" routines to specify options to ARKode.
---------------------------------------------------------------*/

#include <euler3D.hpp>
#include <string.h>

#define MAX_LINE_LENGTH 512


// Load problem-defining parameters from file: root process
// reads parameters and broadcasts results to remaining
// processes
int load_inputs(int myid, double& xl, double& xr, double& yl,
                double& yr, double& zl, double& zr, double& t0,
                double& tf, double& gamma, long int& nx,
                long int& ny, long int& nz, int& xlbc, int& xrbc,
                int& ylbc, int& yrbc, int& zlbc, int& zrbc,
                int& nout, int& showstats)
{
  int retval;
  double dbuff[9];
  long int ibuff[11];

  // root process reads solver parameters from file and packs send buffers
  if (myid == 0) {

    char line[MAX_LINE_LENGTH];
    FILE *FID=NULL;
    FID = fopen("input_euler3D.txt","r");
    if (check_flag((void *) FID, "fopen (load_inputs)", 0)) return(1);
    while (fgets(line, MAX_LINE_LENGTH, FID) != NULL) {

      /* initialize return flag for line */
      retval = 0;

      /* read parameters */
      retval += sscanf(line,"xl = %lf", &xl);
      retval += sscanf(line,"xr = %lf", &xr);
      retval += sscanf(line,"yl = %lf", &yl);
      retval += sscanf(line,"yr = %lf", &yr);
      retval += sscanf(line,"zl = %lf", &zl);
      retval += sscanf(line,"zr = %lf", &zr);
      retval += sscanf(line,"t0 = %lf", &t0);
      retval += sscanf(line,"tf = %lf", &tf);
      retval += sscanf(line,"gamma = %lf", &gamma);
      retval += sscanf(line,"nx = %li", &nx);
      retval += sscanf(line,"ny = %li", &ny);
      retval += sscanf(line,"nz = %li", &nz);
      retval += sscanf(line,"xlbc = %i", &xlbc);
      retval += sscanf(line,"xrbc = %i", &xrbc);
      retval += sscanf(line,"ylbc = %i", &ylbc);
      retval += sscanf(line,"yrbc = %i", &yrbc);
      retval += sscanf(line,"zlbc = %i", &zlbc);
      retval += sscanf(line,"zrbc = %i", &zrbc);
      retval += sscanf(line,"nout = %i", &nout);
      retval += sscanf(line,"showstats = %i", &showstats);

      /* if unable to read the line (and it looks suspicious) issue a warning */
      if (retval == 0 && strstr(line, "=") != NULL && line[0] != '#')
        fprintf(stderr, "load_inputs Warning: parameter line was not interpreted:\n%s", line);

    }
    fclose(FID);

    // pack buffers
    ibuff[0]  = nx;
    ibuff[1]  = ny;
    ibuff[2]  = nz;
    ibuff[3]  = xlbc;
    ibuff[4]  = xrbc;
    ibuff[5]  = ylbc;
    ibuff[6]  = yrbc;
    ibuff[7]  = zlbc;
    ibuff[8]  = zrbc;
    ibuff[9]  = nout;
    ibuff[10] = showstats;
    dbuff[0]  = xl;
    dbuff[1]  = xr;
    dbuff[2]  = yl;
    dbuff[3]  = yr;
    dbuff[4]  = zl;
    dbuff[5]  = zr;
    dbuff[6]  = t0;
    dbuff[7]  = tf;
    dbuff[8]  = gamma;
  }

  // perform broadcast and unpack results
  retval = MPI_Bcast(dbuff, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (load_inputs)", 3)) return(1);
  retval = MPI_Bcast(ibuff, 11, MPI_LONG, 0, MPI_COMM_WORLD);
  if (check_flag(&retval, "MPI_Bcast (load_inputs)", 3)) return(1);

  // unpack buffers
  xl    = dbuff[0];
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
  nout  = ibuff[9];
  showstats = ibuff[10];

  // return with success
  return(0);
}


// Load ARKode solver parameters from file: root process
// reads parameters and broadcasts results to remaining
// processes; all then perform setup
void* arkstep_init_from_file(int myid, const char fname[],
                             const ARKRhsFn f, const ARKRhsFn fe,
                             const ARKRhsFn fi, const realtype t0,
                             const N_Vector w0, int& imex, int& dense_order,
                             int& fxpt, double& rtol, double& atol)
{

  /* declare output */
  void *ark_mem;

  /* declare available solver parameters (with default values) */
  int order = 0;
  int adapt_method = 0;
  int small_nef = 0;
  int msbp = 0;
  int maxcor = 0;
  int predictor = 0;
  int maxnef = 0;
  int maxncf = 0;
  int mxhnil = 0;
  int mxsteps = 0;
  int btable = -1;
  int pq = 0;
  int fixedpt = 0;
  int m_aa = -1;
  double cflfac = 0.0;
  double safety = 0.0;
  double bias = 0.0;
  double growth = 0.0;
  double hfixed_lb = 0.0;
  double hfixed_ub = 0.0;
  double k1 = 0.0;
  double k2 = 0.0;
  double k3 = 0.0;
  double etamx1 = 0.0;
  double etamxf = 0.0;
  double etacf = 0.0;
  double crdown = 0.0;
  double rdiv = 0.0;
  double dgmax = 0.0;
  double nlscoef = 0.0;
  double h0 = 0.0;
  double hmin = 0.0;
  double hmax = 0.0;

  /* all create send/receive buffers read solver parameters from file */
  int ret;
  double dbuff[21];
  int ibuff[16];

  /* root process reads solver parameters from file and packs buffers */
  if (myid == 0) {

    /* open parameter file */
    char line[MAX_LINE_LENGTH];
    FILE *fptr = NULL;
    fptr = fopen(fname,"r");
    if (check_flag((void *) fptr, "fopen (arkstep_init_from_file)", 0)) return(NULL);
    while (fgets(line, MAX_LINE_LENGTH, fptr) != NULL) {

      /* initialize return flag for line */
      ret = 0;

      /* read parameter */
      ret += sscanf(line,"order = %i", &order);
      ret += sscanf(line,"dense_order = %i", &dense_order);
      ret += sscanf(line,"imex = %i", &imex);
      ret += sscanf(line,"btable = %i",  &btable);
      ret += sscanf(line,"adapt_method = %i", &adapt_method);
      ret += sscanf(line,"maxnef = %i", &maxnef);
      ret += sscanf(line,"maxncf = %i", &maxncf);
      ret += sscanf(line,"mxhnil = %i", &mxhnil);
      ret += sscanf(line,"mxsteps = %i", &mxsteps);
      ret += sscanf(line,"cflfac = %lf", &cflfac);
      ret += sscanf(line,"safety = %lf", &safety);
      ret += sscanf(line,"bias = %lf", &bias);
      ret += sscanf(line,"growth = %lf", &growth);
      ret += sscanf(line,"hfixed_lb = %lf", &hfixed_lb);
      ret += sscanf(line,"hfixed_ub = %lf", &hfixed_ub);
      ret += sscanf(line,"pq = %i", &pq);
      ret += sscanf(line,"k1 = %lf", &k1);
      ret += sscanf(line,"k2 = %lf", &k2);
      ret += sscanf(line,"k3 = %lf", &k3);
      ret += sscanf(line,"etamx1 = %lf", &etamx1);
      ret += sscanf(line,"etamxf = %lf", &etamxf);
      ret += sscanf(line,"etacf = %lf", &etacf);
      ret += sscanf(line,"small_nef = %i", &small_nef);
      ret += sscanf(line,"crdown = %lf", &crdown);
      ret += sscanf(line,"rdiv = %lf", &rdiv);
      ret += sscanf(line,"dgmax = %lf", &dgmax);
      ret += sscanf(line,"predictor = %i", &predictor);
      ret += sscanf(line,"msbp = %i", &msbp);
      ret += sscanf(line,"fixedpt = %i", &fixedpt);
      ret += sscanf(line,"m_aa = %i", &m_aa);
      ret += sscanf(line,"maxcor = %i", &maxcor);
      ret += sscanf(line,"nlscoef = %lf", &nlscoef);
      ret += sscanf(line,"h0 = %lf", &h0);
      ret += sscanf(line,"hmin = %lf", &hmin);
      ret += sscanf(line,"hmax = %lf", &hmax);
      ret += sscanf(line,"rtol = %lf", &rtol);
      ret += sscanf(line,"atol = %lf", &atol);

      /* if unable to read the line (and it looks suspicious) issue a warning */
      if (ret == 0 && strstr(line, "=") != NULL && line[0] != '#')
        fprintf(stderr, "arkstep_init_from_file Warning: parameter line was not interpreted:\n%s", line);

    }
    fclose(fptr);

    // pack buffers
    ibuff[0]  = order;
    ibuff[1]  = dense_order;
    ibuff[2]  = imex;
    ibuff[3]  = btable;
    ibuff[4]  = adapt_method;
    ibuff[5]  = maxnef;
    ibuff[6]  = maxncf;
    ibuff[7]  = mxhnil;
    ibuff[8]  = mxsteps;
    ibuff[9]  = pq;
    ibuff[10] = small_nef;
    ibuff[11] = predictor;
    ibuff[12] = msbp;
    ibuff[13] = fixedpt;
    ibuff[14] = m_aa;
    ibuff[15] = maxcor;
    dbuff[0]  = cflfac;
    dbuff[1]  = safety;
    dbuff[2]  = bias;
    dbuff[3]  = growth;
    dbuff[4]  = hfixed_lb;
    dbuff[5]  = hfixed_ub;
    dbuff[6]  = k1;
    dbuff[7]  = k2;
    dbuff[8]  = k3;
    dbuff[9]  = etamx1;
    dbuff[10] = etamxf;
    dbuff[11] = etacf;
    dbuff[12] = crdown;
    dbuff[13] = rdiv;
    dbuff[14] = dgmax;
    dbuff[15] = nlscoef;
    dbuff[16] = h0;
    dbuff[17] = hmin;
    dbuff[18] = hmax;
    dbuff[19] = rtol;
    dbuff[20] = atol;

  }

  // perform broadcast and unpack results
  ret = MPI_Bcast(dbuff, 21, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (check_flag(&ret, "MPI_Bcast (arkstep_init_from_file)", 3)) return(NULL);
  ret = MPI_Bcast(ibuff, 16, MPI_INT, 0, MPI_COMM_WORLD);
  if (check_flag(&ret, "MPI_Bcast (arkstep_init_from_file)", 3)) return(NULL);

  // unpack buffers
  order        = ibuff[0];
  dense_order  = ibuff[1];
  imex         = ibuff[2];
  btable       = ibuff[3];
  adapt_method = ibuff[4];
  maxnef       = ibuff[5];
  maxncf       = ibuff[6];
  mxhnil       = ibuff[7];
  mxsteps      = ibuff[8];
  pq           = ibuff[9];
  small_nef    = ibuff[10];
  predictor    = ibuff[11];
  msbp         = ibuff[12];
  fixedpt      = ibuff[13];
  m_aa         = ibuff[14];
  maxcor       = ibuff[15];
  cflfac       = dbuff[0];
  safety       = dbuff[1];
  bias         = dbuff[2];
  growth       = dbuff[3];
  hfixed_lb    = dbuff[4];
  hfixed_ub    = dbuff[5];
  k1           = dbuff[6];
  k2           = dbuff[7];
  k3           = dbuff[8];
  etamx1       = dbuff[9];
  etamxf       = dbuff[10];
  etacf        = dbuff[11];
  crdown       = dbuff[12];
  rdiv         = dbuff[13];
  dgmax        = dbuff[14];
  nlscoef      = dbuff[15];
  h0           = dbuff[16];
  hmin         = dbuff[17];
  hmax         = dbuff[18];
  rtol         = dbuff[19];
  atol         = dbuff[20];


  /*** check for allowable inputs ***/

  /* check that w0 is not NULL */
  if (w0 == NULL) {
    fprintf(stderr, "arkstep_init_from_file error: cannot initialize problem with w0 == NULL!\n");
    return NULL;
  }

  /* ensure that "imex" agrees with user-supplied rhs functions */
  if ((imex == 2) && (fe == NULL || fi == NULL)) {
    fprintf(stderr, "arkstep_init_from_file error: imex problem but fe or fi is NULL!\n");
    return NULL;
  }
  if ((imex == 0 || imex == 1) && (f == NULL)) {
    fprintf(stderr, "arkstep_init_from_file error: implicit or explicit problem but f is NULL!\n");
    return NULL;
  }



  /*** set outputs to be used by problem ***/
  fxpt = 0;
  if (fixedpt) {
    if (m_aa == 0) fxpt = -1;
    else           fxpt = m_aa;
  }


  /*** Call ARKode routines to initialize integrator and set options ***/

  /* initialize the integrator memory  */
  switch (imex) {
  case 0:         /* purely implicit */
    ark_mem = ARKStepCreate(NULL, f, t0, w0);  break;
  case 1:         /* purely explicit */
    ark_mem = ARKStepCreate(f, NULL, t0, w0);  break;
  default:        /* imex */
    ark_mem = ARKStepCreate(fe, fi, t0, w0);   break;
  }
  if (ark_mem == NULL) {
    fprintf(stderr, "arkstep_init_from_file error in ARKStepCreate\n");
    return NULL;
  }

  /* set RK order, or specify individual Butcher table -- "order" overrides "btable" */
  if (order != 0) {
    ret = ARKStepSetOrder(ark_mem, order);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetOrder = %i\n",ret);
      return NULL;
    }
  } else if (btable != -1) {
    if (imex == 1) {          /* explicit */
      ret = ARKStepSetTableNum(ark_mem, -1, btable);
      if (ret != 0) {
        fprintf(stderr,"arkstep_init_from_file error in ARKStepSetTableNum = %i\n",ret);
        return NULL;
      }
    } else if (imex == 0) {   /* implicit */
      ret = ARKStepSetTableNum(ark_mem, btable, -1);
      if (ret != 0) {
        fprintf(stderr,"arkstep_init_from_file error in ARKStepSetTableNum = %i\n",ret);
        return NULL;
      }
    } else if (imex == 2) {   /* ImEx */
      int btable2;
      if (btable == 2)  btable2 = 104;
      if (btable == 4)  btable2 = 109;
      if (btable == 9)  btable2 = 111;
      if (btable == 13) btable2 = 112;
      if (btable == 14) btable2 = 113;
      ret = ARKStepSetTableNum(ark_mem, btable2, btable);
      if (ret != 0) {
        fprintf(stderr,"arkstep_init_from_file error in ARKStepSetTableNum = %i\n",ret);
        return NULL;
      }
    }
  }

  /* set dense output order */
  ret = ARKStepSetDenseOrder(ark_mem, dense_order);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetDenseOrder = %i\n",ret);
    return NULL;
  }

  /* set cfl stability fraction */
  if (imex != 0) {
    ret = ARKStepSetCFLFraction(ark_mem, cflfac);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetCFLFraction = %i\n",ret);
      return NULL;
    }
  }

  /* set safety factor */
  ret = ARKStepSetSafetyFactor(ark_mem, safety);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetSafetyFactor = %i\n",ret);
    return NULL;
  }

  /* set error bias */
  ret = ARKStepSetErrorBias(ark_mem, bias);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetErrorBias = %i\n",ret);
    return NULL;
  }

  /* set step growth factor */
  ret = ARKStepSetMaxGrowth(ark_mem, growth);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxGrowth = %i\n",ret);
    return NULL;
  }

  /* set fixed step size bounds */
  ret = ARKStepSetFixedStepBounds(ark_mem, hfixed_lb, hfixed_ub);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetFixedStepBounds = %i\n",ret);
    return NULL;
  }

  /* set time step adaptivity method */
  realtype adapt_params[] = {k1, k2, k3};
  int idefault = 1;
  if (fabs(k1)+fabs(k2)+fabs(k3) > 0.0)  idefault=0;
  ret = ARKStepSetAdaptivityMethod(ark_mem, adapt_method, idefault, pq, adapt_params);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetAdaptivityMethod = %i\n",ret);
    return NULL;
  }

  /* set first step growth factor */
  ret = ARKStepSetMaxFirstGrowth(ark_mem, etamx1);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxFirstGrowth = %i\n",ret);
    return NULL;
  }

  /* set error failure growth factor */
  ret = ARKStepSetMaxEFailGrowth(ark_mem, etamxf);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxEFailGrowth = %i\n",ret);
    return NULL;
  }

  /* set number of fails before using above threshold */
  ret = ARKStepSetSmallNumEFails(ark_mem, small_nef);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetSmallNumEFails = %i\n",ret);
    return NULL;
  }

  /* set convergence failure growth factor */
  if (imex != 1) {
    ret = ARKStepSetMaxCFailGrowth(ark_mem, etacf);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxCFailGrowth = %i\n",ret);
      return NULL;
    }
  }

  /* set nonlinear method convergence rate constant */
  if (imex != 1) {
    ret = ARKStepSetNonlinCRDown(ark_mem, crdown);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetNonlinCRDown = %i\n",ret);
      return NULL;
    }
  }

  /* set nonlinear method divergence constant */
  if (imex != 1) {
    ret = ARKStepSetNonlinRDiv(ark_mem, rdiv);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetNonlinRDiv = %i\n",ret);
      return NULL;
    }
  }

  /* set linear solver setup constants */
  if (imex != 1) {
    ret = ARKStepSetDeltaGammaMax(ark_mem, dgmax);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetDeltaGammaMax = %i\n",ret);
      return NULL;
    }
  }

  /* set linear solver setup constants */
  if (imex != 1) {
    ret = ARKStepSetMaxStepsBetweenLSet(ark_mem, msbp);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxStepsBetweenLSet = %i\n",ret);
      return NULL;
    }
  }

  /* set predictor method */
  if (imex != 1) {
    ret = ARKStepSetPredictorMethod(ark_mem, predictor);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetPredictorMethod = %i\n",ret);
      return NULL;
    }
  }

  /* set maximum nonlinear iterations */
  if (imex != 1) {
    ret = ARKStepSetMaxNonlinIters(ark_mem, maxcor);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxNonlinIters = %i\n",ret);
      return NULL;
    }
  }

  /* set nonlinear solver tolerance coefficient */
  if (imex != 1) {
    ret = ARKStepSetNonlinConvCoef(ark_mem, nlscoef);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxNonlinIters = %i\n",ret);
      return NULL;
    }
  }

  /* set initial time step size */
  ret = ARKStepSetInitStep(ark_mem, h0);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetInitStep = %i\n",ret);
    return NULL;
  }

  /* set minimum time step size */
  ret = ARKStepSetMinStep(ark_mem, hmin);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMinStep = %i\n",ret);
    return NULL;
  }

  /* set maximum time step size */
  ret = ARKStepSetMaxStep(ark_mem, hmax);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxStep = %i\n",ret);
    return NULL;
  }

  /* set maximum allowed error test failures */
  ret = ARKStepSetMaxErrTestFails(ark_mem, maxnef);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxErrTestFails = %i\n",ret);
    return NULL;
  }

  /* set maximum allowed convergence failures */
  if (imex != 1) {
    ret = ARKStepSetMaxConvFails(ark_mem, maxncf);
    if (ret != 0) {
      fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxConvFails = %i\n",ret);
      return NULL;
    }
  }

  /* set maximum allowed hnil warnings */
  ret = ARKStepSetMaxHnilWarns(ark_mem, mxhnil);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxHnilWarns = %i\n",ret);
    return NULL;
  }

  /* set maximum allowed steps */
  ret = ARKStepSetMaxNumSteps(ark_mem, mxsteps);
  if (ret != 0) {
    fprintf(stderr,"arkstep_init_from_file error in ARKStepSetMaxNumSteps = %i\n",ret);
    return NULL;
  }

  return ark_mem;
}


/*---- end of file ----*/
