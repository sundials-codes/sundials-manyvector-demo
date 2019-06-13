/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2013, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Utility routine to read input parameters from a specified
 file, and call associated "set" routines to specify options to
 to ARKode solver.
---------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arkode/arkode.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_erkstep.h>
#include <sundials/sundials_types.h>

#define MAX_LINE_LENGTH 512

/* ARKStep version */
void* arkstep_init_from_file(char *fname, ARKRhsFn f, ARKRhsFn fe,
                             ARKRhsFn fi, realtype T0, N_Vector y0,
                             int *ImEx, int *dorder, int *fxpt,
                             realtype *RTol, realtype *ATol) {

  /* declare output */
  void *ark_mem;

  /* declare available solver parameters (with default values) */
  int order = 0;
  int imex = 0;
  int adapt_method = 0;
  int small_nef = 0;
  int msbp = 0;
  int maxcor = 0;
  int predictor = 0;
  int maxnef = 0;
  int maxncf = 0;
  int mxhnil = 0;
  int mxsteps = 0;
  int dense_order = -1;
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
  double rtol = 0.0;
  double atol = 0.0;

  /* open parameter file */
  FILE *fptr = NULL;
  fptr = fopen(fname,"r");
  if (fptr == NULL) {
    fprintf(stderr, "arkstep_init_from_file error: cannot open parameter file %s\n", fname);
    return NULL;
  }

  /* read solver parameters from file */
  int ret;
  char line[MAX_LINE_LENGTH];
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


  /*** check for allowable inputs ***/

  /* check that y0 is not NULL */
  if (y0 == NULL) {
    fprintf(stderr, "arkstep_init_from_file error: cannot initialize problem with y0 == NULL!\n");
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
  *ImEx = imex;
  *dorder = dense_order;
  *RTol = rtol;
  *ATol = atol;
  if (fixedpt) {
       if (m_aa == 0) *fxpt = -1;
    else           *fxpt = m_aa;
  } else {
    *fxpt = 0;
  }


  /*** Call ARKode routines to initialize integrator and set options ***/

  /* initialize the integrator memory  */
  switch (imex) {
  case 0:         /* purely implicit */
    ark_mem = ARKStepCreate(NULL, f, T0, y0);  break;
  case 1:         /* purely explicit */
    ark_mem = ARKStepCreate(f, NULL, T0, y0);  break;
  default:        /* imex */
    ark_mem = ARKStepCreate(fe, fi, T0, y0);   break;
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


/* ERKStep version */
void *erkstep_init_from_file(char *fname, ARKRhsFn f, realtype T0,
                             N_Vector y0, int *dorder,
                             realtype *RTol, realtype *ATol) {

  /* declare output */
  void *ark_mem;

  /* declare available solver parameters (with default values) */
  int order = 0;
  int adapt_method = 0;
  int small_nef = 0;
  int msbp = 0;
  int maxnef = 0;
  int mxhnil = 0;
  int mxsteps = 0;
  int dense_order = -1;
  int btable = -1;
  int pq = 0;
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
  double h0 = 0.0;
  double hmin = 0.0;
  double hmax = 0.0;
  double rtol = 0.0;
  double atol = 0.0;

  /* open parameter file */
  FILE *fptr = NULL;
  fptr = fopen(fname,"r");
  if (fptr == NULL) {
    fprintf(stderr, "erkstep_init_from_file error: cannot open parameter file %s\n", fname);
    return NULL;
  }

  /* read solver parameters from file */
  int ret;
  char line[MAX_LINE_LENGTH];
  while (fgets(line, MAX_LINE_LENGTH, fptr) != NULL) {

    /* initialize return flag for line */
    ret = 0;

    /* read parameter */
    ret += sscanf(line,"order = %i", &order);
    ret += sscanf(line,"dense_order = %i", &dense_order);
    ret += sscanf(line,"btable = %i",  &btable);
    ret += sscanf(line,"adapt_method = %i", &adapt_method);
    ret += sscanf(line,"maxnef = %i", &maxnef);
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
    ret += sscanf(line,"small_nef = %i", &small_nef);
    ret += sscanf(line,"h0 = %lf", &h0);
    ret += sscanf(line,"hmin = %lf", &hmin);
    ret += sscanf(line,"hmax = %lf", &hmax);
    ret += sscanf(line,"rtol = %lf", &rtol);
    ret += sscanf(line,"atol = %lf", &atol);

  }
  fclose(fptr);


  /*** check for allowable inputs ***/

  /* check that y0 is not NULL */
  if (y0 == NULL) {
    fprintf(stderr, "erkstep_init_from_file error: cannot initialize problem with y0 == NULL!\n");
    return NULL;
  }

  /* ensure that RHS function is supplied */
  if (f == NULL) {
    fprintf(stderr, "erkstep_init_from_file error: f is NULL!\n");
    return NULL;
  }


  /*** set outputs to be used by problem ***/
  *dorder = dense_order;
  *RTol = rtol;
  *ATol = atol;


  /*** Call ARKode routines to initialize integrator and set options ***/

  /* initialize the integrator memory  */
  ark_mem = ERKStepCreate(f, T0, y0);
  if (ark_mem == NULL) {
    fprintf(stderr, "arkstep_init_from_file error in ERKStepCreate\n");
    return NULL;
  }

  /* set RK order, or specify individual Butcher table -- "order" overrides "btable" */
  if (order != 0) {     /*  */
    ret = ERKStepSetOrder(ark_mem, order);
    if (ret != 0) {
      fprintf(stderr,"erkstep_init_from_file error in ERKStepSetOrder = %i\n",ret);
      return NULL;
    }
  } else if (btable != -1) {
    ret = ERKStepSetTableNum(ark_mem, btable);
    if (ret != 0) {
      fprintf(stderr,"erkstep_init_from_file error in ERKStepSetTableNum = %i\n",ret);
      return NULL;
    }
  }

  /* set dense output order */
  ret = ERKStepSetDenseOrder(ark_mem, dense_order);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetDenseOrder = %i\n",ret);
    return NULL;
  }

  /* set cfl stability fraction */
  ret = ERKStepSetCFLFraction(ark_mem, cflfac);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetCFLFraction = %i\n",ret);
    return NULL;
  }

  /* set safety factor */
  ret = ERKStepSetSafetyFactor(ark_mem, safety);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetSafetyFactor = %i\n",ret);
    return NULL;
  }

  /* set error bias */
  ret = ERKStepSetErrorBias(ark_mem, bias);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetErrorBias = %i\n",ret);
    return NULL;
  }

  /* set step growth factor */
  ret = ERKStepSetMaxGrowth(ark_mem, growth);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMaxGrowth = %i\n",ret);
    return NULL;
  }

  /* set fixed step size bounds */
  ret = ERKStepSetFixedStepBounds(ark_mem, hfixed_lb, hfixed_ub);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetFixedStepBounds = %i\n",ret);
    return NULL;
  }

  /* set time step adaptivity method */
  realtype adapt_params[] = {k1, k2, k3};
  int idefault = 1;
  if (fabs(k1)+fabs(k2)+fabs(k3) > 0.0)  idefault=0;
  ret = ERKStepSetAdaptivityMethod(ark_mem, adapt_method, idefault, pq, adapt_params);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetAdaptivityMethod = %i\n",ret);
    return NULL;
  }

  /* set first step growth factor */
  ret = ERKStepSetMaxFirstGrowth(ark_mem, etamx1);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMaxFirstGrowth = %i\n",ret);
    return NULL;
  }

  /* set error failure growth factor */
  ret = ERKStepSetMaxEFailGrowth(ark_mem, etamxf);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMaxEFailGrowth = %i\n",ret);
    return NULL;
  }

  /* set number of fails before using above threshold */
  ret = ERKStepSetSmallNumEFails(ark_mem, small_nef);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetSmallNumEFails = %i\n",ret);
    return NULL;
  }

  /* set initial time step size */
  ret = ERKStepSetInitStep(ark_mem, h0);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetInitStep = %i\n",ret);
    return NULL;
  }

  /* set minimum time step size */
  ret = ERKStepSetMinStep(ark_mem, hmin);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMinStep = %i\n",ret);
    return NULL;
  }

  /* set maximum time step size */
  ret = ERKStepSetMaxStep(ark_mem, hmax);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMaxStep = %i\n",ret);
    return NULL;
  }

  /* set maximum allowed error test failures */
  ret = ERKStepSetMaxErrTestFails(ark_mem, maxnef);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMaxErrTestFails = %i\n",ret);
    return NULL;
  }

  /* set maximum allowed hnil warnings */
  ret = ERKStepSetMaxHnilWarns(ark_mem, mxhnil);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMaxHnilWarns = %i\n",ret);
    return NULL;
  }

  /* set maximum allowed steps */
  ret = ERKStepSetMaxNumSteps(ark_mem, mxsteps);
  if (ret != 0) {
    fprintf(stderr,"erkstep_init_from_file error in ERKStepSetMaxNumSteps = %i\n",ret);
    return NULL;
  }

  return ark_mem;
}


/*---- end of file ----*/
