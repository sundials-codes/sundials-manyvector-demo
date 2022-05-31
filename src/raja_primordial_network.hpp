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
#include <mpi.h>
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
  SUNMemory r_k01;
  SUNMemory r_k02;
  SUNMemory r_k03;
  SUNMemory r_k04;
  SUNMemory r_k05;
  SUNMemory r_k06;
  SUNMemory r_k07;
  SUNMemory r_k08;
  SUNMemory r_k09;
  SUNMemory r_k10;
  SUNMemory r_k11;
  SUNMemory r_k12;
  SUNMemory r_k13;
  SUNMemory r_k14;
  SUNMemory r_k15;
  SUNMemory r_k16;
  SUNMemory r_k17;
  SUNMemory r_k18;
  SUNMemory r_k19;
  SUNMemory r_k21;
  SUNMemory r_k22;
  SUNMemory c_brem_brem;
  SUNMemory c_ceHeI_ceHeI;
  SUNMemory c_ceHeII_ceHeII;
  SUNMemory c_ceHI_ceHI;
  SUNMemory c_cie_cooling_cieco;
  SUNMemory c_ciHeI_ciHeI;
  SUNMemory c_ciHeII_ciHeII;
  SUNMemory c_ciHeIS_ciHeIS;
  SUNMemory c_ciHI_ciHI;
  SUNMemory c_compton_comp_;
  SUNMemory c_gloverabel08_gael;
  SUNMemory c_gloverabel08_gaH2;
  SUNMemory c_gloverabel08_gaHe;
  SUNMemory c_gloverabel08_gaHI;
  SUNMemory c_gloverabel08_gaHp;
  SUNMemory c_gloverabel08_h2lte;
  SUNMemory c_h2formation_h2mcool;
  SUNMemory c_h2formation_h2mheat;
  SUNMemory c_h2formation_ncrd1;
  SUNMemory c_h2formation_ncrd2;
  SUNMemory c_h2formation_ncrn;
  SUNMemory c_reHeII1_reHeII1;
  SUNMemory c_reHeII2_reHeII2;
  SUNMemory c_reHeIII_reHeIII;
  SUNMemory c_reHII_reHII;

  // gamma tables
  SUNMemory g_gammaH2_1;
  SUNMemory g_dgammaH2_1_dT;
  SUNMemory g_gammaH2_2;
  SUNMemory g_dgammaH2_2_dT;

  // cell-specific scaling factors
  SUNMemory scale;
  SUNMemory inv_scale;

  // cell-specific reaction rates
  SUNMemory Ts;
  SUNMemory dTs_ge;
  SUNMemory mdensity;
  SUNMemory inv_mdensity;
  SUNMemory rs_k01;
  SUNMemory drs_k01;
  SUNMemory rs_k02;
  SUNMemory drs_k02;
  SUNMemory rs_k03;
  SUNMemory drs_k03;
  SUNMemory rs_k04;
  SUNMemory drs_k04;
  SUNMemory rs_k05;
  SUNMemory drs_k05;
  SUNMemory rs_k06;
  SUNMemory drs_k06;
  SUNMemory rs_k07;
  SUNMemory drs_k07;
  SUNMemory rs_k08;
  SUNMemory drs_k08;
  SUNMemory rs_k09;
  SUNMemory drs_k09;
  SUNMemory rs_k10;
  SUNMemory drs_k10;
  SUNMemory rs_k11;
  SUNMemory drs_k11;
  SUNMemory rs_k12;
  SUNMemory drs_k12;
  SUNMemory rs_k13;
  SUNMemory drs_k13;
  SUNMemory rs_k14;
  SUNMemory drs_k14;
  SUNMemory rs_k15;
  SUNMemory drs_k15;
  SUNMemory rs_k16;
  SUNMemory drs_k16;
  SUNMemory rs_k17;
  SUNMemory drs_k17;
  SUNMemory rs_k18;
  SUNMemory drs_k18;
  SUNMemory rs_k19;
  SUNMemory drs_k19;
  SUNMemory rs_k21;
  SUNMemory drs_k21;
  SUNMemory rs_k22;
  SUNMemory drs_k22;
  SUNMemory cs_brem_brem;
  SUNMemory dcs_brem_brem;
  SUNMemory cs_ceHeI_ceHeI;
  SUNMemory dcs_ceHeI_ceHeI;
  SUNMemory cs_ceHeII_ceHeII;
  SUNMemory dcs_ceHeII_ceHeII;
  SUNMemory cs_ceHI_ceHI;
  SUNMemory dcs_ceHI_ceHI;
  SUNMemory cs_cie_cooling_cieco;
  SUNMemory dcs_cie_cooling_cieco;
  SUNMemory cs_ciHeI_ciHeI;
  SUNMemory dcs_ciHeI_ciHeI;
  SUNMemory cs_ciHeII_ciHeII;
  SUNMemory dcs_ciHeII_ciHeII;
  SUNMemory cs_ciHeIS_ciHeIS;
  SUNMemory dcs_ciHeIS_ciHeIS;
  SUNMemory cs_ciHI_ciHI;
  SUNMemory dcs_ciHI_ciHI;
  SUNMemory cs_compton_comp_;
  SUNMemory dcs_compton_comp_;
  SUNMemory cs_gloverabel08_gael;
  SUNMemory dcs_gloverabel08_gael;
  SUNMemory cs_gloverabel08_gaH2;
  SUNMemory dcs_gloverabel08_gaH2;
  SUNMemory cs_gloverabel08_gaHe;
  SUNMemory dcs_gloverabel08_gaHe;
  SUNMemory cs_gloverabel08_gaHI;
  SUNMemory dcs_gloverabel08_gaHI;
  SUNMemory cs_gloverabel08_gaHp;
  SUNMemory dcs_gloverabel08_gaHp;
  SUNMemory cs_gloverabel08_h2lte;
  SUNMemory dcs_gloverabel08_h2lte;
  SUNMemory cs_h2formation_h2mcool;
  SUNMemory dcs_h2formation_h2mcool;
  SUNMemory cs_h2formation_h2mheat;
  SUNMemory dcs_h2formation_h2mheat;
  SUNMemory cs_h2formation_ncrd1;
  SUNMemory dcs_h2formation_ncrd1;
  SUNMemory cs_h2formation_ncrd2;
  SUNMemory dcs_h2formation_ncrd2;
  SUNMemory cs_h2formation_ncrn;
  SUNMemory dcs_h2formation_ncrn;
  SUNMemory cs_reHeII1_reHeII1;
  SUNMemory dcs_reHeII1_reHeII1;
  SUNMemory cs_reHeII2_reHeII2;
  SUNMemory dcs_reHeII2_reHeII2;
  SUNMemory cs_reHeIII_reHeIII;
  SUNMemory dcs_reHeIII_reHeIII;
  SUNMemory cs_reHII_reHII;
  SUNMemory dcs_reHII_reHII;
  SUNMemory cie_optical_depth_approx;
  SUNMemory h2_optical_depth_approx;

  // strip length
  long int nstrip;

} cvklu_data;


/* Declare ctype RHS and Jacobian */
typedef int(*rhs_f)( realtype, N_Vector , N_Vector , void * );
typedef int(*jac_f)( realtype, N_Vector  , N_Vector , SUNMatrix , void *, N_Vector, N_Vector, N_Vector);

SUNMemory cvklu_setup_data(MPI_Comm, const char *, long int, SUNMemoryHelper, realtype);
void cvklu_free_data(SUNMemory, SUNMemoryHelper);
void cvklu_read_rate_tables(SUNMemory, const char *, int, MPI_Comm);
void cvklu_read_cooling_tables(SUNMemory, const char *, int, MPI_Comm);
void cvklu_read_gamma(SUNMemory, const char *, int, MPI_Comm);
RAJA_DEVICE int cvklu_calculate_temperature(const cvklu_data*, const double*,
                                            const long int, double &, double &);
void setting_up_extra_variables(SUNMemory, long int);

int initialize_sparse_jacobian_cvklu( SUNMatrix J, void *user_data );
int calculate_jacobian_cvklu( realtype t, N_Vector y, N_Vector fy,
                              SUNMatrix J, long int nstrip,
                              void *user_data, N_Vector tmp1,
                              N_Vector tmp2, N_Vector tmp3);
int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot,
                        long int nstrip, void *user_data);


#endif
