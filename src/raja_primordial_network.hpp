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
#include <iostream>
#include "string.h"
#include <RAJA/RAJA.hpp>
#include <sundials/sundials_types.h>
#ifdef USEMAGMA
#include <sunmatrix/sunmatrix_magmadense.h>
#include <sunlinsol/sunlinsol_magmadense.h>
#else
#ifdef RAJA_CUDA
#include <sunmatrix/sunmatrix_cusparse.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#elif RAJA_SERIAL
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>
#else
#error RAJA HIP chemistry interface is currently unimplemented
#endif
#endif

#define NSPECIES 10


// desired execution policy for all RAJA loops
#ifdef RAJA_SERIAL
using EXECPOLICY = RAJA::loop_exec;
using REDUCEPOLICY = RAJA::loop_reduce;
using XYZ_KERNEL_POL =
  RAJA::KernelPolicy< RAJA::statement::For<2, EXECPOLICY,
                        RAJA::statement::For<1, EXECPOLICY,
                          RAJA::statement::For<0, EXECPOLICY,
                            RAJA::statement::Lambda<0>
                          >
                        >
                      >
                    >;
#elif RAJA_CUDA
using EXECPOLICY = RAJA::cuda_exec<256>;
using REDUCEPOLICY = RAJA::cuda_reduce;
using XYZ_KERNEL_POL =
  RAJA::KernelPolicy< RAJA::statement::CudaKernel<
                          RAJA::statement::For<2, RAJA::cuda_thread_x_loop,
                            RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
                              RAJA::statement::For<0, RAJA::cuda_thread_z_loop,
                                RAJA::statement::Lambda<0>
                              >
                            >
                          >
                        >
                      >;
#else // RAJA_HIP
#define EXECPOLICY    RAJA::hip_exec<256>
#define REDUCEPOLICY  RAJA::hip_reduce
using XYZ_KERNEL_POL =
    RAJA::KernelPolicy< RAJA::HipKernel<
                          RAJA::statement::For<2, RAJA::hip_thread_x_loop,
                            RAJA::statement::For<1, RAJA::hip_thread_y_loop,
                              RAJA::statement::For<0, RAJA::hip_thread_z_loop,
                                RAJA::statement::Lambda<0>
                              >
                            >
                          >
                        >
                      >;
#endif



typedef struct cvklu_data {
  /* All of the network bins will be the same width */
  double dbin;
  double idbin;
  double bounds[2];
  int nbins;

  /* For storing and passing around redshift information */
  double current_z;

  /* Cooling and chemical tables */
  double* r_k01;
  double* r_k02;
  double* r_k03;
  double* r_k04;
  double* r_k05;
  double* r_k06;
  double* r_k07;
  double* r_k08;
  double* r_k09;
  double* r_k10;
  double* r_k11;
  double* r_k12;
  double* r_k13;
  double* r_k14;
  double* r_k15;
  double* r_k16;
  double* r_k17;
  double* r_k18;
  double* r_k19;
  double* r_k21;
  double* r_k22;
  double* c_brem_brem;
  double* c_ceHeI_ceHeI;
  double* c_ceHeII_ceHeII;
  double* c_ceHI_ceHI;
  double* c_cie_cooling_cieco;
  double* c_ciHeI_ciHeI;
  double* c_ciHeII_ciHeII;
  double* c_ciHeIS_ciHeIS;
  double* c_ciHI_ciHI;
  double* c_compton_comp_;
  double* c_gloverabel08_gael;
  double* c_gloverabel08_gaH2;
  double* c_gloverabel08_gaHe;
  double* c_gloverabel08_gaHI;
  double* c_gloverabel08_gaHp;
  double* c_gloverabel08_h2lte;
  double* c_h2formation_h2mcool;
  double* c_h2formation_h2mheat;
  double* c_h2formation_ncrd1;
  double* c_h2formation_ncrd2;
  double* c_h2formation_ncrn;
  double* c_reHeII1_reHeII1;
  double* c_reHeII2_reHeII2;
  double* c_reHeIII_reHeIII;
  double* c_reHII_reHII;

  // gamma tables
  double* g_gammaH2_1;
  double* g_dgammaH2_1_dT;
  double* g_gammaH2_2;
  double* g_dgammaH2_2_dT;

  // cell-specific scaling factors
  double *scale;
  double *inv_scale;

  // cell-specific reaction rates
  double *Ts;
  double *dTs_ge;
  double *mdensity;
  double *inv_mdensity;
  double *rs_k01;
  double *drs_k01;
  double *rs_k02;
  double *drs_k02;
  double *rs_k03;
  double *drs_k03;
  double *rs_k04;
  double *drs_k04;
  double *rs_k05;
  double *drs_k05;
  double *rs_k06;
  double *drs_k06;
  double *rs_k07;
  double *drs_k07;
  double *rs_k08;
  double *drs_k08;
  double *rs_k09;
  double *drs_k09;
  double *rs_k10;
  double *drs_k10;
  double *rs_k11;
  double *drs_k11;
  double *rs_k12;
  double *drs_k12;
  double *rs_k13;
  double *drs_k13;
  double *rs_k14;
  double *drs_k14;
  double *rs_k15;
  double *drs_k15;
  double *rs_k16;
  double *drs_k16;
  double *rs_k17;
  double *drs_k17;
  double *rs_k18;
  double *drs_k18;
  double *rs_k19;
  double *drs_k19;
  double *rs_k21;
  double *drs_k21;
  double *rs_k22;
  double *drs_k22;
  double *cs_brem_brem;
  double *dcs_brem_brem;
  double *cs_ceHeI_ceHeI;
  double *dcs_ceHeI_ceHeI;
  double *cs_ceHeII_ceHeII;
  double *dcs_ceHeII_ceHeII;
  double *cs_ceHI_ceHI;
  double *dcs_ceHI_ceHI;
  double *cs_cie_cooling_cieco;
  double *dcs_cie_cooling_cieco;
  double *cs_ciHeI_ciHeI;
  double *dcs_ciHeI_ciHeI;
  double *cs_ciHeII_ciHeII;
  double *dcs_ciHeII_ciHeII;
  double *cs_ciHeIS_ciHeIS;
  double *dcs_ciHeIS_ciHeIS;
  double *cs_ciHI_ciHI;
  double *dcs_ciHI_ciHI;
  double *cs_compton_comp_;
  double *dcs_compton_comp_;
  double *cs_gloverabel08_gael;
  double *dcs_gloverabel08_gael;
  double *cs_gloverabel08_gaH2;
  double *dcs_gloverabel08_gaH2;
  double *cs_gloverabel08_gaHe;
  double *dcs_gloverabel08_gaHe;
  double *cs_gloverabel08_gaHI;
  double *dcs_gloverabel08_gaHI;
  double *cs_gloverabel08_gaHp;
  double *dcs_gloverabel08_gaHp;
  double *cs_gloverabel08_h2lte;
  double *dcs_gloverabel08_h2lte;
  double *cs_h2formation_h2mcool;
  double *dcs_h2formation_h2mcool;
  double *cs_h2formation_h2mheat;
  double *dcs_h2formation_h2mheat;
  double *cs_h2formation_ncrd1;
  double *dcs_h2formation_ncrd1;
  double *cs_h2formation_ncrd2;
  double *dcs_h2formation_ncrd2;
  double *cs_h2formation_ncrn;
  double *dcs_h2formation_ncrn;
  double *cs_reHeII1_reHeII1;
  double *dcs_reHeII1_reHeII1;
  double *cs_reHeII2_reHeII2;
  double *dcs_reHeII2_reHeII2;
  double *cs_reHeIII_reHeIII;
  double *dcs_reHeIII_reHeIII;
  double *cs_reHII_reHII;
  double *dcs_reHII_reHII;
  double *cie_optical_depth_approx;
  double *h2_optical_depth_approx;

  // strip length
  long int nstrip;

  const char *dengo_data_file;
} cvklu_data;


/* Declare ctype RHS and Jacobian */
typedef int(*rhs_f)( realtype, N_Vector , N_Vector , void * );
typedef int(*jac_f)( realtype, N_Vector  , N_Vector , SUNMatrix , void *, N_Vector, N_Vector, N_Vector);

cvklu_data *cvklu_setup_data(const char *, long int, SUNMemoryHelper, realtype);
void cvklu_free_data(void*, SUNMemoryHelper);
void cvklu_read_rate_tables(cvklu_data*, const char *, int);
void cvklu_read_cooling_tables(cvklu_data*, const char *, int);
void cvklu_read_gamma(cvklu_data*, const char *, int);
RAJA_DEVICE int cvklu_calculate_temperature(const cvklu_data*, const double*,
                                            const long int, double &, double &);
void setting_up_extra_variables(cvklu_data*, long int);

int initialize_sparse_jacobian_cvklu( SUNMatrix J, void *user_data );
int calculate_jacobian_cvklu( realtype t, N_Vector y, N_Vector fy,
                              SUNMatrix J, long int nstrip,
                              void *user_data, N_Vector tmp1,
                              N_Vector tmp2, N_Vector tmp3);
int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot,
                        long int nstrip, void *user_data);


#endif
