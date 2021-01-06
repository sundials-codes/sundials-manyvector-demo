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


/* stdlib, hdf5, local includes */

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

/* header files for SUNDIALS */
#include <nvector/nvector_serial.h>      /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h>   /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h>   /* access to dense SUNLinearSolver */
#include <sundials/sundials_types.h>     /* defs. of realtype, sunindextype */
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunlinsol/sunlinsol_klu.h>


/* User-defined vector and matrix accessor macros: Ith, IJth */

/* These macros are defined in order to write code which exactly matches
   the mathematical problem description given above.

   Ith(v,i) references the ith component of the vector v, where i is in
   the range [1..NEQ] and NEQ is defined below. The Ith macro is defined
   using the N_VIth macro in nvector.h. N_VIth numbers the components of
   a vector starting from 0.

   IJth(A,i,j) references the (i,j)th element of the dense matrix A, where
   i and j are in the range [1..NEQ]. The IJth macro is defined using the
   DENSE_ELEM macro in dense.h. DENSE_ELEM numbers rows and columns of a
   dense matrix starting from 0. */

#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */


#ifndef MAX_NCELLS
#define MAX_NCELLS 1024
#endif

#define NSPECIES 10
#define DMAX(A,B) ((A) > (B) ? (A) : (B))
#define DMIN(A,B) ((A) < (B) ? (A) : (B))



typedef struct cvklu_data {
    /* All of the network bins will be the same width */
    double dbin;
    double idbin;
    double bounds[2];
    int nbins;

    /* These will be for bins in redshift space */
    double d_zbin;
    double id_zbin;
    double z_bounds[2];
    int n_zbins;

    /* For storing and passing around
       redshift information */
    double current_z;
    double zdef;
    double dz;

    double Ts[1][MAX_NCELLS];
    double Tdef[1][MAX_NCELLS]; /* t1, t2, tdef */
    double dT[1][MAX_NCELLS]; /* t1, t2, tdef */
    double logTs[1][MAX_NCELLS];
    double invTs[1][MAX_NCELLS];
    double dTs_ge[1][MAX_NCELLS];

    /* Now we do all of our cooling and chemical tables */
    double r_k01[1024];
    double rs_k01[1][MAX_NCELLS];
    double drs_k01[1][MAX_NCELLS];

    double r_k02[1024];
    double rs_k02[1][MAX_NCELLS];
    double drs_k02[1][MAX_NCELLS];

    double r_k03[1024];
    double rs_k03[1][MAX_NCELLS];
    double drs_k03[1][MAX_NCELLS];

    double r_k04[1024];
    double rs_k04[1][MAX_NCELLS];
    double drs_k04[1][MAX_NCELLS];

    double r_k05[1024];
    double rs_k05[1][MAX_NCELLS];
    double drs_k05[1][MAX_NCELLS];

    double r_k06[1024];
    double rs_k06[1][MAX_NCELLS];
    double drs_k06[1][MAX_NCELLS];

    double r_k07[1024];
    double rs_k07[1][MAX_NCELLS];
    double drs_k07[1][MAX_NCELLS];

    double r_k08[1024];
    double rs_k08[1][MAX_NCELLS];
    double drs_k08[1][MAX_NCELLS];

    double r_k09[1024];
    double rs_k09[1][MAX_NCELLS];
    double drs_k09[1][MAX_NCELLS];

    double r_k10[1024];
    double rs_k10[1][MAX_NCELLS];
    double drs_k10[1][MAX_NCELLS];

    double r_k11[1024];
    double rs_k11[1][MAX_NCELLS];
    double drs_k11[1][MAX_NCELLS];

    double r_k12[1024];
    double rs_k12[1][MAX_NCELLS];
    double drs_k12[1][MAX_NCELLS];

    double r_k13[1024];
    double rs_k13[1][MAX_NCELLS];
    double drs_k13[1][MAX_NCELLS];

    double r_k14[1024];
    double rs_k14[1][MAX_NCELLS];
    double drs_k14[1][MAX_NCELLS];

    double r_k15[1024];
    double rs_k15[1][MAX_NCELLS];
    double drs_k15[1][MAX_NCELLS];

    double r_k16[1024];
    double rs_k16[1][MAX_NCELLS];
    double drs_k16[1][MAX_NCELLS];

    double r_k17[1024];
    double rs_k17[1][MAX_NCELLS];
    double drs_k17[1][MAX_NCELLS];

    double r_k18[1024];
    double rs_k18[1][MAX_NCELLS];
    double drs_k18[1][MAX_NCELLS];

    double r_k19[1024];
    double rs_k19[1][MAX_NCELLS];
    double drs_k19[1][MAX_NCELLS];

    double r_k21[1024];
    double rs_k21[1][MAX_NCELLS];
    double drs_k21[1][MAX_NCELLS];

    double r_k22[1024];
    double rs_k22[1][MAX_NCELLS];
    double drs_k22[1][MAX_NCELLS];

    double c_brem_brem[1024];
    double cs_brem_brem[1][MAX_NCELLS];
    double dcs_brem_brem[1][MAX_NCELLS];

    double c_ceHeI_ceHeI[1024];
    double cs_ceHeI_ceHeI[1][MAX_NCELLS];
    double dcs_ceHeI_ceHeI[1][MAX_NCELLS];

    double c_ceHeII_ceHeII[1024];
    double cs_ceHeII_ceHeII[1][MAX_NCELLS];
    double dcs_ceHeII_ceHeII[1][MAX_NCELLS];

    double c_ceHI_ceHI[1024];
    double cs_ceHI_ceHI[1][MAX_NCELLS];
    double dcs_ceHI_ceHI[1][MAX_NCELLS];

    double c_cie_cooling_cieco[1024];
    double cs_cie_cooling_cieco[1][MAX_NCELLS];
    double dcs_cie_cooling_cieco[1][MAX_NCELLS];

    double c_ciHeI_ciHeI[1024];
    double cs_ciHeI_ciHeI[1][MAX_NCELLS];
    double dcs_ciHeI_ciHeI[1][MAX_NCELLS];

    double c_ciHeII_ciHeII[1024];
    double cs_ciHeII_ciHeII[1][MAX_NCELLS];
    double dcs_ciHeII_ciHeII[1][MAX_NCELLS];

    double c_ciHeIS_ciHeIS[1024];
    double cs_ciHeIS_ciHeIS[1][MAX_NCELLS];
    double dcs_ciHeIS_ciHeIS[1][MAX_NCELLS];

    double c_ciHI_ciHI[1024];
    double cs_ciHI_ciHI[1][MAX_NCELLS];
    double dcs_ciHI_ciHI[1][MAX_NCELLS];

    double c_compton_comp_[1024];
    double cs_compton_comp_[1][MAX_NCELLS];
    double dcs_compton_comp_[1][MAX_NCELLS];

    double c_gammah_gammah[1024];
    double cs_gammah_gammah[1][MAX_NCELLS];
    double dcs_gammah_gammah[1][MAX_NCELLS];

    double c_gloverabel08_gael[1024];
    double cs_gloverabel08_gael[1][MAX_NCELLS];
    double dcs_gloverabel08_gael[1][MAX_NCELLS];
    double c_gloverabel08_gaH2[1024];
    double cs_gloverabel08_gaH2[1][MAX_NCELLS];
    double dcs_gloverabel08_gaH2[1][MAX_NCELLS];
    double c_gloverabel08_gaHe[1024];
    double cs_gloverabel08_gaHe[1][MAX_NCELLS];
    double dcs_gloverabel08_gaHe[1][MAX_NCELLS];
    double c_gloverabel08_gaHI[1024];
    double cs_gloverabel08_gaHI[1][MAX_NCELLS];
    double dcs_gloverabel08_gaHI[1][MAX_NCELLS];
    double c_gloverabel08_gaHp[1024];
    double cs_gloverabel08_gaHp[1][MAX_NCELLS];
    double dcs_gloverabel08_gaHp[1][MAX_NCELLS];
    double c_gloverabel08_gphdl[1024];
    double cs_gloverabel08_gphdl[1][MAX_NCELLS];
    double dcs_gloverabel08_gphdl[1][MAX_NCELLS];
    double c_gloverabel08_gpldl[1024];
    double cs_gloverabel08_gpldl[1][MAX_NCELLS];
    double dcs_gloverabel08_gpldl[1][MAX_NCELLS];
    double c_gloverabel08_h2lte[1024];
    double cs_gloverabel08_h2lte[1][MAX_NCELLS];
    double dcs_gloverabel08_h2lte[1][MAX_NCELLS];

    double c_h2formation_h2mcool[1024];
    double cs_h2formation_h2mcool[1][MAX_NCELLS];
    double dcs_h2formation_h2mcool[1][MAX_NCELLS];
    double c_h2formation_h2mheat[1024];
    double cs_h2formation_h2mheat[1][MAX_NCELLS];
    double dcs_h2formation_h2mheat[1][MAX_NCELLS];
    double c_h2formation_ncrd1[1024];
    double cs_h2formation_ncrd1[1][MAX_NCELLS];
    double dcs_h2formation_ncrd1[1][MAX_NCELLS];
    double c_h2formation_ncrd2[1024];
    double cs_h2formation_ncrd2[1][MAX_NCELLS];
    double dcs_h2formation_ncrd2[1][MAX_NCELLS];
    double c_h2formation_ncrn[1024];
    double cs_h2formation_ncrn[1][MAX_NCELLS];
    double dcs_h2formation_ncrn[1][MAX_NCELLS];

    double c_reHeII1_reHeII1[1024];
    double cs_reHeII1_reHeII1[1][MAX_NCELLS];
    double dcs_reHeII1_reHeII1[1][MAX_NCELLS];

    double c_reHeII2_reHeII2[1024];
    double cs_reHeII2_reHeII2[1][MAX_NCELLS];
    double dcs_reHeII2_reHeII2[1][MAX_NCELLS];

    double c_reHeIII_reHeIII[1024];
    double cs_reHeIII_reHeIII[1][MAX_NCELLS];
    double dcs_reHeIII_reHeIII[1][MAX_NCELLS];

    double c_reHII_reHII[1024];
    double cs_reHII_reHII[1][MAX_NCELLS];
    double dcs_reHII_reHII[1][MAX_NCELLS];

    int bin_id[1][MAX_NCELLS];
    int ncells;



    // gamma as a function of temperature
    double g_gammaH2_1[1024];
    double g_dgammaH2_1_dT[1024];

    // store the gamma for that particular step
    double gammaH2_1[1][MAX_NCELLS];
    double dgammaH2_1_dT[1][MAX_NCELLS];

    // store 1 / (gamma - 1)
    double _gammaH2_1_m1[1][MAX_NCELLS];

    double g_gammaH2_2[1024];
    double g_dgammaH2_2_dT[1024];

    // store the gamma for that particular step
    double gammaH2_2[1][MAX_NCELLS];
    double dgammaH2_2_dT[1][MAX_NCELLS];

    // store 1 / (gamma - 1)
    double _gammaH2_2_m1[1][MAX_NCELLS];


    double scale[1][10 * MAX_NCELLS ];
    double inv_scale[1][10 * MAX_NCELLS];


    int nstrip;
    double mdensity[1][MAX_NCELLS];
    double inv_mdensity[1][MAX_NCELLS];


    double cie_optical_depth_approx[1][MAX_NCELLS];


    double h2_optical_depth_approx[1][MAX_NCELLS];


    const char *dengo_data_file;
} cvklu_data;


/* Declare ctype RHS and Jacobian */
typedef int(*rhs_f)( realtype, N_Vector , N_Vector , void * );
typedef int(*jac_f)( realtype, N_Vector  , N_Vector , SUNMatrix , void *, N_Vector, N_Vector, N_Vector);

cvklu_data *cvklu_setup_data(const char *, int *, char***);
void cvklu_read_rate_tables(cvklu_data*);
void cvklu_read_cooling_tables(cvklu_data*);
void cvklu_read_gamma(cvklu_data*);
void cvklu_interpolate_gamma(cvklu_data*, int );

void setting_up_extra_variables( cvklu_data * data, double * input, int nstrip );






int calculate_sparse_jacobian_cvklu( realtype t,
                                        N_Vector y, N_Vector fy,
                                        SUNMatrix J, void *user_data,
                                        N_Vector tmp1, N_Vector tmp2,
                                        N_Vector tmp3);

int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot, void *user_data);
void ensure_electron_consistency(double *input, int nstrip, int nchem);
void temperature_from_mass_density(double *input, int nstrip, int nchem,
                                   double *strip_temperature);

int cvklu_calculate_temperature(cvklu_data *data, double *input, int nstrip, int nchem);




typedef struct dengo_field_data
{

  unsigned long int nstrip;
  unsigned long int ncells;
  // This should be updated dynamically
  // with dengo
  double *density;
  double *H2_1_density;

  double *H2_2_density;

  double *H_1_density;

  double *H_2_density;

  double *H_m0_density;

  double *He_1_density;

  double *He_2_density;

  double *He_3_density;

  double *de_density;

  double *ge_density;


  double *CoolingTime;
  double *MolecularWeight;
  double *temperature;
  double *Gamma;

  const char *dengo_data_file;
} dengo_field_data;

typedef struct code_units
{

  int comoving_coordinates;
  double density_units;
  double length_units;
  double time_units;
  double velocity_units;
  double a_units;
  double a_value;

} code_units;


#endif
