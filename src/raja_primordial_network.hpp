/*---------------------------------------------------------------
  Programmer(s): Daniel R. Reynolds @ SMU
  ----------------------------------------------------------------
  Copyright (c) 2021, Southern Methodist University.
  All rights reserved.
  For details, see the LICENSE file.
  ----------------------------------------------------------------
  Header file for RAJA port of Dengo-based primordial chemistry
  network.
  ---------------------------------------------------------------*/

// only include this file once (if included multiple times)
#ifndef __RAJA_PRIMORDIAL_HPP__
#define __RAJA_PRIMORDIAL_HPP__


// Header files
#include "time.h"
#include "sys/time.h"
#include "stdlib.h"
#include "math.h"
#ifdef USEHDF5
#include "hdf5.h"
#include "hdf5_hl.h"
#endif
#include "stdio.h"
#include "string.h"
#include <RAJA/RAJA.hpp>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>

#define NSPECIES 10


// desired execution policy for all RAJA loops
#define EXECPOLICY RAJA::loop_exec


typedef struct cell_rate_data {
  double Ts;
  double dTs_ge;
  double mdensity;
  double inv_mdensity;
  double rs_k01;
  double drs_k01;
  double rs_k02;
  double drs_k02;
  double rs_k03;
  double drs_k03;
  double rs_k04;
  double drs_k04;
  double rs_k05;
  double drs_k05;
  double rs_k06;
  double drs_k06;
  double rs_k07;
  double drs_k07;
  double rs_k08;
  double drs_k08;
  double rs_k09;
  double drs_k09;
  double rs_k10;
  double drs_k10;
  double rs_k11;
  double drs_k11;
  double rs_k12;
  double drs_k12;
  double rs_k13;
  double drs_k13;
  double rs_k14;
  double drs_k14;
  double rs_k15;
  double drs_k15;
  double rs_k16;
  double drs_k16;
  double rs_k17;
  double drs_k17;
  double rs_k18;
  double drs_k18;
  double rs_k19;
  double drs_k19;
  double rs_k21;
  double drs_k21;
  double rs_k22;
  double drs_k22;
  double cs_brem_brem;
  double dcs_brem_brem;
  double cs_ceHeI_ceHeI;
  double dcs_ceHeI_ceHeI;
  double cs_ceHeII_ceHeII;
  double dcs_ceHeII_ceHeII;
  double cs_ceHI_ceHI;
  double dcs_ceHI_ceHI;
  double cs_cie_cooling_cieco;
  double dcs_cie_cooling_cieco;
  double cs_ciHeI_ciHeI;
  double dcs_ciHeI_ciHeI;
  double cs_ciHeII_ciHeII;
  double dcs_ciHeII_ciHeII;
  double cs_ciHeIS_ciHeIS;
  double dcs_ciHeIS_ciHeIS;
  double cs_ciHI_ciHI;
  double dcs_ciHI_ciHI;
  double cs_compton_comp_;
  double dcs_compton_comp_;
  double cs_gloverabel08_gael;
  double dcs_gloverabel08_gael;
  double cs_gloverabel08_gaH2;
  double dcs_gloverabel08_gaH2;
  double cs_gloverabel08_gaHe;
  double dcs_gloverabel08_gaHe;
  double cs_gloverabel08_gaHI;
  double dcs_gloverabel08_gaHI;
  double cs_gloverabel08_gaHp;
  double dcs_gloverabel08_gaHp;
  double cs_gloverabel08_h2lte;
  double dcs_gloverabel08_h2lte;
  double cs_h2formation_h2mcool;
  double dcs_h2formation_h2mcool;
  double cs_h2formation_h2mheat;
  double dcs_h2formation_h2mheat;
  double cs_h2formation_ncrd1;
  double dcs_h2formation_ncrd1;
  double cs_h2formation_ncrd2;
  double dcs_h2formation_ncrd2;
  double cs_h2formation_ncrn;
  double dcs_h2formation_ncrn;
  double cs_reHeII1_reHeII1;
  double dcs_reHeII1_reHeII1;
  double cs_reHeII2_reHeII2;
  double dcs_reHeII2_reHeII2;
  double cs_reHeIII_reHeIII;
  double dcs_reHeIII_reHeIII;
  double cs_reHII_reHII;
  double dcs_reHII_reHII;
  double gammaH2_1;
  double dgammaH2_1_dT;
  double gammaH2_2;
  double dgammaH2_2_dT;
  double cie_optical_depth_approx;
  double h2_optical_depth_approx;
} cell_rate_data;


typedef struct cvklu_data {
  /* All of the network bins will be the same width */
  double dbin;
  double idbin;
  double bounds[2];
  int nbins;

  /* For storing and passing around redshift information */
  double current_z;

  /* Cell-specific derived quantities reaction rate information */
  cell_rate_data *cell_data;

  /* Cooling and chemical tables */
  double r_k01[1024];
  double r_k02[1024];
  double r_k03[1024];
  double r_k04[1024];
  double r_k05[1024];
  double r_k06[1024];
  double r_k07[1024];
  double r_k08[1024];
  double r_k09[1024];
  double r_k10[1024];
  double r_k11[1024];
  double r_k12[1024];
  double r_k13[1024];
  double r_k14[1024];
  double r_k15[1024];
  double r_k16[1024];
  double r_k17[1024];
  double r_k18[1024];
  double r_k19[1024];
  double r_k21[1024];
  double r_k22[1024];
  double c_brem_brem[1024];
  double c_ceHeI_ceHeI[1024];
  double c_ceHeII_ceHeII[1024];
  double c_ceHI_ceHI[1024];
  double c_cie_cooling_cieco[1024];
  double c_ciHeI_ciHeI[1024];
  double c_ciHeII_ciHeII[1024];
  double c_ciHeIS_ciHeIS[1024];
  double c_ciHI_ciHI[1024];
  double c_compton_comp_[1024];
  double c_gloverabel08_gael[1024];
  double c_gloverabel08_gaH2[1024];
  double c_gloverabel08_gaHe[1024];
  double c_gloverabel08_gaHI[1024];
  double c_gloverabel08_gaHp[1024];
  double c_gloverabel08_h2lte[1024];
  double c_h2formation_h2mcool[1024];
  double c_h2formation_h2mheat[1024];
  double c_h2formation_ncrd1[1024];
  double c_h2formation_ncrd2[1024];
  double c_h2formation_ncrn[1024];
  double c_reHeII1_reHeII1[1024];
  double c_reHeII2_reHeII2[1024];
  double c_reHeIII_reHeIII[1024];
  double c_reHII_reHII[1024];

  // gamma as a function of temperature
  double g_gammaH2_1[1024];
  double g_dgammaH2_1_dT[1024];
  double g_gammaH2_2[1024];
  double g_dgammaH2_2_dT[1024];

  // scaling factors
  double *scale;
  double *inv_scale;

  int nstrip;

  const char *dengo_data_file;
} cvklu_data;


/* Declare ctype RHS and Jacobian */
typedef int(*rhs_f)( realtype, N_Vector , N_Vector , void * );
typedef int(*jac_f)( realtype, N_Vector  , N_Vector , SUNMatrix , void *, N_Vector, N_Vector, N_Vector);

cvklu_data *cvklu_setup_data(const char *, int);
void cvklu_free_data(void*);
void cvklu_read_rate_tables(cvklu_data*);
void cvklu_read_cooling_tables(cvklu_data*);
void cvklu_read_gamma(cvklu_data*);
void cvklu_interpolate_rates(cvklu_data*, cell_rate_data&);
void cvklu_interpolate_gamma(cvklu_data*, cell_rate_data&);
int cvklu_calculate_temperature(cvklu_data*, double*, cell_rate_data&);
void setting_up_extra_variables(cvklu_data*, double*, int);

int calculate_sparse_jacobian_cvklu( realtype t,
                                     N_Vector y, N_Vector fy,
                                     SUNMatrix J, void *user_data,
                                     N_Vector tmp1, N_Vector tmp2,
                                     N_Vector tmp3);
int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot, void *user_data);


#endif
