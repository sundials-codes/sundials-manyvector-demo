/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2021, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Implementation file for RAJA port of Dengo-based primordial
 chemistry network.
 ---------------------------------------------------------------*/

#include <raja_primordial_network.hpp>


cvklu_data *cvklu_setup_data(const char *FileLocation, long int ncells)
{

  //-----------------------------------------------------
  // Function : cvklu_setup_data
  // Description: Initialize a data object that stores the reaction/ cooling rate data
  //-----------------------------------------------------

  cvklu_data *data = (cvklu_data *) malloc(sizeof(cvklu_data));

  // point the module to look for cvklu_tables.h5
  data->dengo_data_file = FileLocation;

  // Number of cells to be solved in a batch
  data->nstrip = ncells;

  // allocate reaction rate arrays
#ifdef RAJA_SERIAL
  data->Ts = (double *) malloc(ncells*sizeof(double));
  data->dTs_ge = (double *) malloc(ncells*sizeof(double));
  data->mdensity = (double *) malloc(ncells*sizeof(double));
  data->inv_mdensity = (double *) malloc(ncells*sizeof(double));
  data->rs_k01 = (double *) malloc(ncells*sizeof(double));
  data->drs_k01 = (double *) malloc(ncells*sizeof(double));
  data->rs_k02 = (double *) malloc(ncells*sizeof(double));
  data->drs_k02 = (double *) malloc(ncells*sizeof(double));
  data->rs_k03 = (double *) malloc(ncells*sizeof(double));
  data->drs_k03 = (double *) malloc(ncells*sizeof(double));
  data->rs_k04 = (double *) malloc(ncells*sizeof(double));
  data->drs_k04 = (double *) malloc(ncells*sizeof(double));
  data->rs_k05 = (double *) malloc(ncells*sizeof(double));
  data->drs_k05 = (double *) malloc(ncells*sizeof(double));
  data->rs_k06 = (double *) malloc(ncells*sizeof(double));
  data->drs_k06 = (double *) malloc(ncells*sizeof(double));
  data->rs_k07 = (double *) malloc(ncells*sizeof(double));
  data->drs_k07 = (double *) malloc(ncells*sizeof(double));
  data->rs_k08 = (double *) malloc(ncells*sizeof(double));
  data->drs_k08 = (double *) malloc(ncells*sizeof(double));
  data->rs_k09 = (double *) malloc(ncells*sizeof(double));
  data->drs_k09 = (double *) malloc(ncells*sizeof(double));
  data->rs_k10 = (double *) malloc(ncells*sizeof(double));
  data->drs_k10 = (double *) malloc(ncells*sizeof(double));
  data->rs_k11 = (double *) malloc(ncells*sizeof(double));
  data->drs_k11 = (double *) malloc(ncells*sizeof(double));
  data->rs_k12 = (double *) malloc(ncells*sizeof(double));
  data->drs_k12 = (double *) malloc(ncells*sizeof(double));
  data->rs_k13 = (double *) malloc(ncells*sizeof(double));
  data->drs_k13 = (double *) malloc(ncells*sizeof(double));
  data->rs_k14 = (double *) malloc(ncells*sizeof(double));
  data->drs_k14 = (double *) malloc(ncells*sizeof(double));
  data->rs_k15 = (double *) malloc(ncells*sizeof(double));
  data->drs_k15 = (double *) malloc(ncells*sizeof(double));
  data->rs_k16 = (double *) malloc(ncells*sizeof(double));
  data->drs_k16 = (double *) malloc(ncells*sizeof(double));
  data->rs_k17 = (double *) malloc(ncells*sizeof(double));
  data->drs_k17 = (double *) malloc(ncells*sizeof(double));
  data->rs_k18 = (double *) malloc(ncells*sizeof(double));
  data->drs_k18 = (double *) malloc(ncells*sizeof(double));
  data->rs_k19 = (double *) malloc(ncells*sizeof(double));
  data->drs_k19 = (double *) malloc(ncells*sizeof(double));
  data->rs_k21 = (double *) malloc(ncells*sizeof(double));
  data->drs_k21 = (double *) malloc(ncells*sizeof(double));
  data->rs_k22 = (double *) malloc(ncells*sizeof(double));
  data->drs_k22 = (double *) malloc(ncells*sizeof(double));
  data->cs_brem_brem = (double *) malloc(ncells*sizeof(double));
  data->dcs_brem_brem = (double *) malloc(ncells*sizeof(double));
  data->cs_ceHeI_ceHeI = (double *) malloc(ncells*sizeof(double));
  data->dcs_ceHeI_ceHeI = (double *) malloc(ncells*sizeof(double));
  data->cs_ceHeII_ceHeII = (double *) malloc(ncells*sizeof(double));
  data->dcs_ceHeII_ceHeII = (double *) malloc(ncells*sizeof(double));
  data->cs_ceHI_ceHI = (double *) malloc(ncells*sizeof(double));
  data->dcs_ceHI_ceHI = (double *) malloc(ncells*sizeof(double));
  data->cs_cie_cooling_cieco = (double *) malloc(ncells*sizeof(double));
  data->dcs_cie_cooling_cieco = (double *) malloc(ncells*sizeof(double));
  data->cs_ciHeI_ciHeI = (double *) malloc(ncells*sizeof(double));
  data->dcs_ciHeI_ciHeI = (double *) malloc(ncells*sizeof(double));
  data->cs_ciHeII_ciHeII = (double *) malloc(ncells*sizeof(double));
  data->dcs_ciHeII_ciHeII = (double *) malloc(ncells*sizeof(double));
  data->cs_ciHeIS_ciHeIS = (double *) malloc(ncells*sizeof(double));
  data->dcs_ciHeIS_ciHeIS = (double *) malloc(ncells*sizeof(double));
  data->cs_ciHI_ciHI = (double *) malloc(ncells*sizeof(double));
  data->dcs_ciHI_ciHI = (double *) malloc(ncells*sizeof(double));
  data->cs_compton_comp_ = (double *) malloc(ncells*sizeof(double));
  data->dcs_compton_comp_ = (double *) malloc(ncells*sizeof(double));
  data->cs_gloverabel08_gael = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_gael = (double *) malloc(ncells*sizeof(double));
  data->cs_gloverabel08_gaH2 = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_gaH2 = (double *) malloc(ncells*sizeof(double));
  data->cs_gloverabel08_gaHe = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_gaHe = (double *) malloc(ncells*sizeof(double));
  data->cs_gloverabel08_gaHI = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_gaHI = (double *) malloc(ncells*sizeof(double));
  data->cs_gloverabel08_gaHp = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_gaHp = (double *) malloc(ncells*sizeof(double));
  data->cs_gloverabel08_h2lte = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_h2lte = (double *) malloc(ncells*sizeof(double));
  data->cs_h2formation_h2mcool = (double *) malloc(ncells*sizeof(double));
  data->dcs_h2formation_h2mcool = (double *) malloc(ncells*sizeof(double));
  data->cs_h2formation_h2mheat = (double *) malloc(ncells*sizeof(double));
  data->dcs_h2formation_h2mheat = (double *) malloc(ncells*sizeof(double));
  data->cs_h2formation_ncrd1 = (double *) malloc(ncells*sizeof(double));
  data->dcs_h2formation_ncrd1 = (double *) malloc(ncells*sizeof(double));
  data->cs_h2formation_ncrd2 = (double *) malloc(ncells*sizeof(double));
  data->dcs_h2formation_ncrd2 = (double *) malloc(ncells*sizeof(double));
  data->cs_h2formation_ncrn = (double *) malloc(ncells*sizeof(double));
  data->dcs_h2formation_ncrn = (double *) malloc(ncells*sizeof(double));
  data->cs_reHeII1_reHeII1 = (double *) malloc(ncells*sizeof(double));
  data->dcs_reHeII1_reHeII1 = (double *) malloc(ncells*sizeof(double));
  data->cs_reHeII2_reHeII2 = (double *) malloc(ncells*sizeof(double));
  data->dcs_reHeII2_reHeII2 = (double *) malloc(ncells*sizeof(double));
  data->cs_reHeIII_reHeIII = (double *) malloc(ncells*sizeof(double));
  data->dcs_reHeIII_reHeIII = (double *) malloc(ncells*sizeof(double));
  data->cs_reHII_reHII = (double *) malloc(ncells*sizeof(double));
  data->dcs_reHII_reHII = (double *) malloc(ncells*sizeof(double));
  data->gammaH2_1 = (double *) malloc(ncells*sizeof(double));
  data->dgammaH2_1_dT = (double *) malloc(ncells*sizeof(double));
  data->gammaH2_2 = (double *) malloc(ncells*sizeof(double));
  data->dgammaH2_2_dT = (double *) malloc(ncells*sizeof(double));
  data->cie_optical_depth_approx = (double *) malloc(ncells*sizeof(double));
  data->h2_optical_depth_approx = (double *) malloc(ncells*sizeof(double));
#elif RAJA_CUDA
  cudaMallocManaged((void**)&(data->Ts), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dTs_ge), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->mdensity), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->inv_mdensity), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k01), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k01), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k02), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k02), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k03), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k03), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k04), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k04), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k05), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k05), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k06), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k06), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k07), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k07), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k08), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k08), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k09), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k09), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k10), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k10), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k11), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k11), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k12), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k12), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k13), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k13), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k14), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k14), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k15), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k15), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k16), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k16), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k17), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k17), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k18), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k18), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k19), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k19), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k21), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k21), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->rs_k22), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->drs_k22), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_brem_brem), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_brem_brem), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_ceHeI_ceHeI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_ceHeI_ceHeI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_ceHeII_ceHeII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_ceHeII_ceHeII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_ceHI_ceHI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_ceHI_ceHI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_cie_cooling_cieco), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_cie_cooling_cieco), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_ciHeI_ciHeI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_ciHeI_ciHeI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_ciHeII_ciHeII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_ciHeII_ciHeII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_ciHeIS_ciHeIS), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_ciHeIS_ciHeIS), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_ciHI_ciHI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_ciHI_ciHI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_compton_comp_), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_compton_comp_), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_gloverabel08_gael), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_gloverabel08_gael), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_gloverabel08_gaH2), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_gloverabel08_gaH2), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_gloverabel08_gaHe), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_gloverabel08_gaHe), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_gloverabel08_gaHI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_gloverabel08_gaHI), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_gloverabel08_gaHp), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_gloverabel08_gaHp), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_gloverabel08_h2lte), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_gloverabel08_h2lte), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_h2formation_h2mcool), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_h2formation_h2mcool), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_h2formation_h2mheat), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_h2formation_h2mheat), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_h2formation_ncrd1), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_h2formation_ncrd1), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_h2formation_ncrd2), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_h2formation_ncrd2), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_h2formation_ncrn), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_h2formation_ncrn), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_reHeII1_reHeII1), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_reHeII1_reHeII1), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_reHeII2_reHeII2), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_reHeII2_reHeII2), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_reHeIII_reHeIII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_reHeIII_reHeIII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cs_reHII_reHII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dcs_reHII_reHII), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->gammaH2_1), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dgammaH2_1_dT), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->gammaH2_2), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->dgammaH2_2_dT), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->cie_optical_depth_approx), ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->h2_optical_depth_approx), ncells*sizeof(double));
#else
#error RAJA HIP chemistry interface is currently unimplemented
#endif

  // allocate scaling arrays
#ifdef RAJA_SERIAL
  data->scale = (double *) malloc(NSPECIES*ncells*sizeof(double));
  data->inv_scale = (double *) malloc(NSPECIES*ncells*sizeof(double));
#elif RAJA_CUDA
  cudaMallocManaged((void**)&(data->scale), NSPECIES*ncells*sizeof(double));
  cudaMallocManaged((void**)&(data->inv_scale), NSPECIES*ncells*sizeof(double));
#else
#error RAJA HIP chemistry interface is currently unimplemented
#endif
  data->current_z = 0.0;

  // initialize temperature so it wont crash
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,ncells), [=] RAJA_DEVICE (long int i) {
    data->Ts[i] = 1000.0;
  });

  // Temperature-related pieces
  data->bounds[0] = 1.0;
  data->bounds[1] = 100000.0;
  data->nbins = 1024 - 1;
  data->dbin = (log(data->bounds[1]) - log(data->bounds[0])) / data->nbins;
  data->idbin = 1.0L / data->dbin;

  cvklu_read_rate_tables(data);
  cvklu_read_cooling_tables(data);
  cvklu_read_gamma(data);

  data->dengo_data_file = NULL;

  return data;

}


void cvklu_free_data(void *data)
{

  //-----------------------------------------------------
  // Function : cvklu_free_data
  // Description: Frees reaction/ cooling rate data
  //-----------------------------------------------------
  cvklu_data *rxdata = (cvklu_data *) data;

#ifdef RAJA_SERIAL
  free(rxdata->scale);
  free(rxdata->inv_scale);
  free(rxdata->Ts);
  free(rxdata->dTs_ge);
  free(rxdata->mdensity);
  free(rxdata->inv_mdensity);
  free(rxdata->rs_k01);
  free(rxdata->drs_k01);
  free(rxdata->rs_k02);
  free(rxdata->drs_k02);
  free(rxdata->rs_k03);
  free(rxdata->drs_k03);
  free(rxdata->rs_k04);
  free(rxdata->drs_k04);
  free(rxdata->rs_k05);
  free(rxdata->drs_k05);
  free(rxdata->rs_k06);
  free(rxdata->drs_k06);
  free(rxdata->rs_k07);
  free(rxdata->drs_k07);
  free(rxdata->rs_k08);
  free(rxdata->drs_k08);
  free(rxdata->rs_k09);
  free(rxdata->drs_k09);
  free(rxdata->rs_k10);
  free(rxdata->drs_k10);
  free(rxdata->rs_k11);
  free(rxdata->drs_k11);
  free(rxdata->rs_k12);
  free(rxdata->drs_k12);
  free(rxdata->rs_k13);
  free(rxdata->drs_k13);
  free(rxdata->rs_k14);
  free(rxdata->drs_k14);
  free(rxdata->rs_k15);
  free(rxdata->drs_k15);
  free(rxdata->rs_k16);
  free(rxdata->drs_k16);
  free(rxdata->rs_k17);
  free(rxdata->drs_k17);
  free(rxdata->rs_k18);
  free(rxdata->drs_k18);
  free(rxdata->rs_k19);
  free(rxdata->drs_k19);
  free(rxdata->rs_k21);
  free(rxdata->drs_k21);
  free(rxdata->rs_k22);
  free(rxdata->drs_k22);
  free(rxdata->cs_brem_brem);
  free(rxdata->dcs_brem_brem);
  free(rxdata->cs_ceHeI_ceHeI);
  free(rxdata->dcs_ceHeI_ceHeI);
  free(rxdata->cs_ceHeII_ceHeII);
  free(rxdata->dcs_ceHeII_ceHeII);
  free(rxdata->cs_ceHI_ceHI);
  free(rxdata->dcs_ceHI_ceHI);
  free(rxdata->cs_cie_cooling_cieco);
  free(rxdata->dcs_cie_cooling_cieco);
  free(rxdata->cs_ciHeI_ciHeI);
  free(rxdata->dcs_ciHeI_ciHeI);
  free(rxdata->cs_ciHeII_ciHeII);
  free(rxdata->dcs_ciHeII_ciHeII);
  free(rxdata->cs_ciHeIS_ciHeIS);
  free(rxdata->dcs_ciHeIS_ciHeIS);
  free(rxdata->cs_ciHI_ciHI);
  free(rxdata->dcs_ciHI_ciHI);
  free(rxdata->cs_compton_comp_);
  free(rxdata->dcs_compton_comp_);
  free(rxdata->cs_gloverabel08_gael);
  free(rxdata->dcs_gloverabel08_gael);
  free(rxdata->cs_gloverabel08_gaH2);
  free(rxdata->dcs_gloverabel08_gaH2);
  free(rxdata->cs_gloverabel08_gaHe);
  free(rxdata->dcs_gloverabel08_gaHe);
  free(rxdata->cs_gloverabel08_gaHI);
  free(rxdata->dcs_gloverabel08_gaHI);
  free(rxdata->cs_gloverabel08_gaHp);
  free(rxdata->dcs_gloverabel08_gaHp);
  free(rxdata->cs_gloverabel08_h2lte);
  free(rxdata->dcs_gloverabel08_h2lte);
  free(rxdata->cs_h2formation_h2mcool);
  free(rxdata->dcs_h2formation_h2mcool);
  free(rxdata->cs_h2formation_h2mheat);
  free(rxdata->dcs_h2formation_h2mheat);
  free(rxdata->cs_h2formation_ncrd1);
  free(rxdata->dcs_h2formation_ncrd1);
  free(rxdata->cs_h2formation_ncrd2);
  free(rxdata->dcs_h2formation_ncrd2);
  free(rxdata->cs_h2formation_ncrn);
  free(rxdata->dcs_h2formation_ncrn);
  free(rxdata->cs_reHeII1_reHeII1);
  free(rxdata->dcs_reHeII1_reHeII1);
  free(rxdata->cs_reHeII2_reHeII2);
  free(rxdata->dcs_reHeII2_reHeII2);
  free(rxdata->cs_reHeIII_reHeIII);
  free(rxdata->dcs_reHeIII_reHeIII);
  free(rxdata->cs_reHII_reHII);
  free(rxdata->dcs_reHII_reHII);
  free(rxdata->gammaH2_1);
  free(rxdata->dgammaH2_1_dT);
  free(rxdata->gammaH2_2);
  free(rxdata->dgammaH2_2_dT);
  free(rxdata->cie_optical_depth_approx);
  free(rxdata->h2_optical_depth_approx);
#elif RAJA_CUDA
  cudaFree(rxdata->scale);
  cudaFree(rxdata->inv_scale);
  cudaFree(rxdata->Ts);
  cudaFree(rxdata->dTs_ge);
  cudaFree(rxdata->mdensity);
  cudaFree(rxdata->inv_mdensity);
  cudaFree(rxdata->rs_k01);
  cudaFree(rxdata->drs_k01);
  cudaFree(rxdata->rs_k02);
  cudaFree(rxdata->drs_k02);
  cudaFree(rxdata->rs_k03);
  cudaFree(rxdata->drs_k03);
  cudaFree(rxdata->rs_k04);
  cudaFree(rxdata->drs_k04);
  cudaFree(rxdata->rs_k05);
  cudaFree(rxdata->drs_k05);
  cudaFree(rxdata->rs_k06);
  cudaFree(rxdata->drs_k06);
  cudaFree(rxdata->rs_k07);
  cudaFree(rxdata->drs_k07);
  cudaFree(rxdata->rs_k08);
  cudaFree(rxdata->drs_k08);
  cudaFree(rxdata->rs_k09);
  cudaFree(rxdata->drs_k09);
  cudaFree(rxdata->rs_k10);
  cudaFree(rxdata->drs_k10);
  cudaFree(rxdata->rs_k11);
  cudaFree(rxdata->drs_k11);
  cudaFree(rxdata->rs_k12);
  cudaFree(rxdata->drs_k12);
  cudaFree(rxdata->rs_k13);
  cudaFree(rxdata->drs_k13);
  cudaFree(rxdata->rs_k14);
  cudaFree(rxdata->drs_k14);
  cudaFree(rxdata->rs_k15);
  cudaFree(rxdata->drs_k15);
  cudaFree(rxdata->rs_k16);
  cudaFree(rxdata->drs_k16);
  cudaFree(rxdata->rs_k17);
  cudaFree(rxdata->drs_k17);
  cudaFree(rxdata->rs_k18);
  cudaFree(rxdata->drs_k18);
  cudaFree(rxdata->rs_k19);
  cudaFree(rxdata->drs_k19);
  cudaFree(rxdata->rs_k21);
  cudaFree(rxdata->drs_k21);
  cudaFree(rxdata->rs_k22);
  cudaFree(rxdata->drs_k22);
  cudaFree(rxdata->cs_brem_brem);
  cudaFree(rxdata->dcs_brem_brem);
  cudaFree(rxdata->cs_ceHeI_ceHeI);
  cudaFree(rxdata->dcs_ceHeI_ceHeI);
  cudaFree(rxdata->cs_ceHeII_ceHeII);
  cudaFree(rxdata->dcs_ceHeII_ceHeII);
  cudaFree(rxdata->cs_ceHI_ceHI);
  cudaFree(rxdata->dcs_ceHI_ceHI);
  cudaFree(rxdata->cs_cie_cooling_cieco);
  cudaFree(rxdata->dcs_cie_cooling_cieco);
  cudaFree(rxdata->cs_ciHeI_ciHeI);
  cudaFree(rxdata->dcs_ciHeI_ciHeI);
  cudaFree(rxdata->cs_ciHeII_ciHeII);
  cudaFree(rxdata->dcs_ciHeII_ciHeII);
  cudaFree(rxdata->cs_ciHeIS_ciHeIS);
  cudaFree(rxdata->dcs_ciHeIS_ciHeIS);
  cudaFree(rxdata->cs_ciHI_ciHI);
  cudaFree(rxdata->dcs_ciHI_ciHI);
  cudaFree(rxdata->cs_compton_comp_);
  cudaFree(rxdata->dcs_compton_comp_);
  cudaFree(rxdata->cs_gloverabel08_gael);
  cudaFree(rxdata->dcs_gloverabel08_gael);
  cudaFree(rxdata->cs_gloverabel08_gaH2);
  cudaFree(rxdata->dcs_gloverabel08_gaH2);
  cudaFree(rxdata->cs_gloverabel08_gaHe);
  cudaFree(rxdata->dcs_gloverabel08_gaHe);
  cudaFree(rxdata->cs_gloverabel08_gaHI);
  cudaFree(rxdata->dcs_gloverabel08_gaHI);
  cudaFree(rxdata->cs_gloverabel08_gaHp);
  cudaFree(rxdata->dcs_gloverabel08_gaHp);
  cudaFree(rxdata->cs_gloverabel08_h2lte);
  cudaFree(rxdata->dcs_gloverabel08_h2lte);
  cudaFree(rxdata->cs_h2formation_h2mcool);
  cudaFree(rxdata->dcs_h2formation_h2mcool);
  cudaFree(rxdata->cs_h2formation_h2mheat);
  cudaFree(rxdata->dcs_h2formation_h2mheat);
  cudaFree(rxdata->cs_h2formation_ncrd1);
  cudaFree(rxdata->dcs_h2formation_ncrd1);
  cudaFree(rxdata->cs_h2formation_ncrd2);
  cudaFree(rxdata->dcs_h2formation_ncrd2);
  cudaFree(rxdata->cs_h2formation_ncrn);
  cudaFree(rxdata->dcs_h2formation_ncrn);
  cudaFree(rxdata->cs_reHeII1_reHeII1);
  cudaFree(rxdata->dcs_reHeII1_reHeII1);
  cudaFree(rxdata->cs_reHeII2_reHeII2);
  cudaFree(rxdata->dcs_reHeII2_reHeII2);
  cudaFree(rxdata->cs_reHeIII_reHeIII);
  cudaFree(rxdata->dcs_reHeIII_reHeIII);
  cudaFree(rxdata->cs_reHII_reHII);
  cudaFree(rxdata->dcs_reHII_reHII);
  cudaFree(rxdata->gammaH2_1);
  cudaFree(rxdata->dgammaH2_1_dT);
  cudaFree(rxdata->gammaH2_2);
  cudaFree(rxdata->dgammaH2_2_dT);
  cudaFree(rxdata->cie_optical_depth_approx);
  cudaFree(rxdata->h2_optical_depth_approx);
#else
#error RAJA HIP chemistry interface is currently unimplemented
#endif
  free(rxdata);

}


void cvklu_read_rate_tables(cvklu_data *data)
{
  const char * filedir;
  if (data->dengo_data_file != NULL){
    filedir =  data->dengo_data_file;
  } else{
    filedir = "cvklu_tables.h5";
  }

  hid_t file_id = H5Fopen( filedir , H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read the rate tables (these go to main memory)
  H5LTread_dataset_double(file_id, "/k01", data->r_k01);
  H5LTread_dataset_double(file_id, "/k02", data->r_k02);
  H5LTread_dataset_double(file_id, "/k03", data->r_k03);
  H5LTread_dataset_double(file_id, "/k04", data->r_k04);
  H5LTread_dataset_double(file_id, "/k05", data->r_k05);
  H5LTread_dataset_double(file_id, "/k06", data->r_k06);
  H5LTread_dataset_double(file_id, "/k07", data->r_k07);
  H5LTread_dataset_double(file_id, "/k08", data->r_k08);
  H5LTread_dataset_double(file_id, "/k09", data->r_k09);
  H5LTread_dataset_double(file_id, "/k10", data->r_k10);
  H5LTread_dataset_double(file_id, "/k11", data->r_k11);
  H5LTread_dataset_double(file_id, "/k12", data->r_k12);
  H5LTread_dataset_double(file_id, "/k13", data->r_k13);
  H5LTread_dataset_double(file_id, "/k14", data->r_k14);
  H5LTread_dataset_double(file_id, "/k15", data->r_k15);
  H5LTread_dataset_double(file_id, "/k16", data->r_k16);
  H5LTread_dataset_double(file_id, "/k17", data->r_k17);
  H5LTread_dataset_double(file_id, "/k18", data->r_k18);
  H5LTread_dataset_double(file_id, "/k19", data->r_k19);
  H5LTread_dataset_double(file_id, "/k21", data->r_k21);
  H5LTread_dataset_double(file_id, "/k22", data->r_k22);
  H5Fclose(file_id);

  // ensure that rate tables are synchronized to device memory
#ifdef RAJA_CUDA
  cudaDeviceSynchronize();
#elif RAJA_HIP
#error RAJA HIP chemistry interface is currently unimplemented
#endif
}


void cvklu_read_cooling_tables(cvklu_data *data)
{

  const char * filedir;
  if (data->dengo_data_file != NULL){
    filedir =  data->dengo_data_file;
  } else{
    filedir = "cvklu_tables.h5";
  }
  hid_t file_id = H5Fopen( filedir , H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read the cooling tables (these go to main memory)
  H5LTread_dataset_double(file_id, "/brem_brem",           data->c_brem_brem);
  H5LTread_dataset_double(file_id, "/ceHeI_ceHeI",         data->c_ceHeI_ceHeI);
  H5LTread_dataset_double(file_id, "/ceHeII_ceHeII",       data->c_ceHeII_ceHeII);
  H5LTread_dataset_double(file_id, "/ceHI_ceHI",           data->c_ceHI_ceHI);
  H5LTread_dataset_double(file_id, "/cie_cooling_cieco",   data->c_cie_cooling_cieco);
  H5LTread_dataset_double(file_id, "/ciHeI_ciHeI",         data->c_ciHeI_ciHeI);
  H5LTread_dataset_double(file_id, "/ciHeII_ciHeII",       data->c_ciHeII_ciHeII);
  H5LTread_dataset_double(file_id, "/ciHeIS_ciHeIS",       data->c_ciHeIS_ciHeIS);
  H5LTread_dataset_double(file_id, "/ciHI_ciHI",           data->c_ciHI_ciHI);
  H5LTread_dataset_double(file_id, "/compton_comp_",       data->c_compton_comp_);
  H5LTread_dataset_double(file_id, "/gloverabel08_gael",   data->c_gloverabel08_gael);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaH2",   data->c_gloverabel08_gaH2);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHe",   data->c_gloverabel08_gaHe);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHI",   data->c_gloverabel08_gaHI);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHp",   data->c_gloverabel08_gaHp);
  H5LTread_dataset_double(file_id, "/gloverabel08_h2lte",  data->c_gloverabel08_h2lte);
  H5LTread_dataset_double(file_id, "/h2formation_h2mcool", data->c_h2formation_h2mcool);
  H5LTread_dataset_double(file_id, "/h2formation_h2mheat", data->c_h2formation_h2mheat);
  H5LTread_dataset_double(file_id, "/h2formation_ncrd1",   data->c_h2formation_ncrd1);
  H5LTread_dataset_double(file_id, "/h2formation_ncrd2",   data->c_h2formation_ncrd2);
  H5LTread_dataset_double(file_id, "/h2formation_ncrn",    data->c_h2formation_ncrn);
  H5LTread_dataset_double(file_id, "/reHeII1_reHeII1",     data->c_reHeII1_reHeII1);
  H5LTread_dataset_double(file_id, "/reHeII2_reHeII2",     data->c_reHeII2_reHeII2);
  H5LTread_dataset_double(file_id, "/reHeIII_reHeIII",     data->c_reHeIII_reHeIII);
  H5LTread_dataset_double(file_id, "/reHII_reHII",         data->c_reHII_reHII);
  H5Fclose(file_id);

  // ensure that cooling tables are synchronized to device memory
#ifdef RAJA_CUDA
  cudaDeviceSynchronize();
#elif RAJA_HIP
#error RAJA HIP chemistry interface is currently unimplemented
#endif
}

void cvklu_read_gamma(cvklu_data *data)
{

  const char * filedir;
  if (data->dengo_data_file != NULL){
    filedir =  data->dengo_data_file;
  } else{
    filedir = "cvklu_tables.h5";
  }

  hid_t file_id = H5Fopen( filedir , H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read the gamma tables (these go to main memory)
  H5LTread_dataset_double(file_id, "/gammaH2_1",     data->g_gammaH2_1 );
  H5LTread_dataset_double(file_id, "/dgammaH2_1_dT", data->g_dgammaH2_1_dT );
  H5LTread_dataset_double(file_id, "/gammaH2_2",     data->g_gammaH2_2 );
  H5LTread_dataset_double(file_id, "/dgammaH2_2_dT", data->g_dgammaH2_2_dT );
  H5Fclose(file_id);

  // ensure that gamma tables are synchronized to device memory
#ifdef RAJA_CUDA
  cudaDeviceSynchronize();
#elif RAJA_HIP
#error RAJA HIP chemistry interface is currently unimplemented
#endif
}



RAJA_DEVICE int cvklu_calculate_temperature(cvklu_data *data, double *input, long int i)
{

  // Define some constants
  const double kb = 1.3806504e-16; // Boltzmann constant [erg/K]
  const double mh = 1.67e-24;
  const double gamma = 5.e0/3.e0;
  const double _gamma_m1 = 1.0 / (gamma - 1);
  //const int MAX_T_ITERATION = 100;

  // Calculate total density
  double H2_1 = input[0];
  double H2_2 = input[1];
  double H_1 = input[2];
  double H_2 = input[3];
  double H_m0 = input[4];
  double He_1 = input[5];
  double He_2 = input[6];
  double He_3 = input[7];
  double de = input[8];
  double ge = input[9];
  double density = 2.0*H2_1 + 2.0*H2_2 + 1.0079400000000001*H_1
    + 1.0079400000000001*H_2 + 1.0079400000000001*H_m0 + 4.0026020000000004*He_1
    + 4.0026020000000004*He_2 + 4.0026020000000004*He_3;

  // Initiate the "guess" temperature
  double T    = data->Ts[i];
  double Tnew = T*1.1;
  double Tdiff = Tnew - T;
  double dge_dT;
  //int count = 0;

  // We do Newton's Iteration to calculate the temperature
  // Since gammaH2 is dependent on the temperature too!
  //        while ( Tdiff/ Tnew > 0.001 ){
  for (int j=0; j<10; j++){

    T = data->Ts[i];
    cvklu_interpolate_gamma(data, i);

    double gammaH2_1 = data->gammaH2_1[i];
    double dgammaH2_1_dT = data->dgammaH2_1_dT[i];
    double _gammaH2_1_m1 = 1.0 / (gammaH2_1 - 1.0);

    double gammaH2_2 = data->gammaH2_2[i];
    double dgammaH2_2_dT = data->dgammaH2_2_dT[i];
    double _gammaH2_2_m1 = 1.0 / (gammaH2_2 - 1.0);

    // update gammaH2
    // The derivatives of  sum (nkT/(gamma - 1)/mh/density) - ge
    // This is the function we want to minimize
    // which should only be dependent on the first part
    dge_dT = T*kb*(-H2_1*_gammaH2_1_m1*_gammaH2_1_m1*dgammaH2_1_dT - H2_2*_gammaH2_2_m1*_gammaH2_2_m1*dgammaH2_2_dT)/(density*mh) + kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh);

    //This is the change in ge for each iteration
    double dge = T*kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh) - ge;

    Tnew = T - dge/dge_dT;
    data->Ts[i] = Tnew;

    Tdiff = fabs(T - Tnew);
    // fprintf(stderr, "T: %0.5g ; Tnew: %0.5g; dge_dT: %.5g, dge: %.5g, ge: %.5g \n", T,Tnew, dge_dT, dge, ge);
    // count += 1;
    // if (count > MAX_T_ITERATION){
    //     fprintf(stderr, "T failed to converge \n");
    //     return 1;
    // }
  } // while loop

  data->Ts[i] = Tnew;

  if (data->Ts[i] < data->bounds[0]) {
    data->Ts[i] = data->bounds[0];
  } else if (data->Ts[i] > data->bounds[1]) {
    data->Ts[i] = data->bounds[1];
  }
  data->dTs_ge[i] = 1.0 / dge_dT;

  return 0;

}



RAJA_DEVICE void cvklu_interpolate_rates(cvklu_data *data, long int i)
{
  int bin_id;
  double lb, t1, t2;
  double Tdef, dT, invTs, Tfactor;

  lb = log(data->bounds[0]);

  bin_id = (int) (data->idbin * (log(data->Ts[i]) - lb));
  if (bin_id <= 0) {
    bin_id = 0;
  } else if (bin_id >= data->nbins) {
    bin_id = data->nbins - 1;
  }
  t1 = (lb + (bin_id    ) * data->dbin);
  t2 = (lb + (bin_id + 1) * data->dbin);
  Tdef = (log(data->Ts[i]) - t1)/(t2 - t1);
  dT = (t2 - t1);
  invTs = 1.0 / data->Ts[i];
  Tfactor = invTs/dT;

  data->rs_k01[i] = data->r_k01[bin_id] +
    Tdef * (data->r_k01[bin_id+1] - data->r_k01[bin_id]);
  data->drs_k01[i] = (data->r_k01[bin_id+1] - data->r_k01[bin_id])*Tfactor;

  data->rs_k02[i] = data->r_k02[bin_id] +
    Tdef * (data->r_k02[bin_id+1] - data->r_k02[bin_id]);
  data->drs_k02[i] = (data->r_k02[bin_id+1] - data->r_k02[bin_id])*Tfactor;

  data->rs_k03[i] = data->r_k03[bin_id] +
    Tdef * (data->r_k03[bin_id+1] - data->r_k03[bin_id]);
  data->drs_k03[i] = (data->r_k03[bin_id+1] - data->r_k03[bin_id])*Tfactor;

  data->rs_k04[i] = data->r_k04[bin_id] +
    Tdef * (data->r_k04[bin_id+1] - data->r_k04[bin_id]);
  data->drs_k04[i] = (data->r_k04[bin_id+1] - data->r_k04[bin_id])*Tfactor;

  data->rs_k05[i] = data->r_k05[bin_id] +
    Tdef * (data->r_k05[bin_id+1] - data->r_k05[bin_id]);
  data->drs_k05[i] = (data->r_k05[bin_id+1] - data->r_k05[bin_id])*Tfactor;

  data->rs_k06[i] = data->r_k06[bin_id] +
    Tdef * (data->r_k06[bin_id+1] - data->r_k06[bin_id]);
  data->drs_k06[i] = (data->r_k06[bin_id+1] - data->r_k06[bin_id])*Tfactor;

  data->rs_k07[i] = data->r_k07[bin_id] +
    Tdef * (data->r_k07[bin_id+1] - data->r_k07[bin_id]);
  data->drs_k07[i] = (data->r_k07[bin_id+1] - data->r_k07[bin_id])*Tfactor;

  data->rs_k08[i] = data->r_k08[bin_id] +
    Tdef * (data->r_k08[bin_id+1] - data->r_k08[bin_id]);
  data->drs_k08[i] = (data->r_k08[bin_id+1] - data->r_k08[bin_id])*Tfactor;

  data->rs_k09[i] = data->r_k09[bin_id] +
    Tdef * (data->r_k09[bin_id+1] - data->r_k09[bin_id]);
  data->drs_k09[i] = (data->r_k09[bin_id+1] - data->r_k09[bin_id])*Tfactor;

  data->rs_k10[i] = data->r_k10[bin_id] +
    Tdef * (data->r_k10[bin_id+1] - data->r_k10[bin_id]);
  data->drs_k10[i] = (data->r_k10[bin_id+1] - data->r_k10[bin_id])*Tfactor;

  data->rs_k11[i] = data->r_k11[bin_id] +
    Tdef * (data->r_k11[bin_id+1] - data->r_k11[bin_id]);
  data->drs_k11[i] = (data->r_k11[bin_id+1] - data->r_k11[bin_id])*Tfactor;

  data->rs_k12[i] = data->r_k12[bin_id] +
    Tdef * (data->r_k12[bin_id+1] - data->r_k12[bin_id]);
  data->drs_k12[i] = (data->r_k12[bin_id+1] - data->r_k12[bin_id])*Tfactor;

  data->rs_k13[i] = data->r_k13[bin_id] +
    Tdef * (data->r_k13[bin_id+1] - data->r_k13[bin_id]);
  data->drs_k13[i] = (data->r_k13[bin_id+1] - data->r_k13[bin_id])*Tfactor;

  data->rs_k14[i] = data->r_k14[bin_id] +
    Tdef * (data->r_k14[bin_id+1] - data->r_k14[bin_id]);
  data->drs_k14[i] = (data->r_k14[bin_id+1] - data->r_k14[bin_id])*Tfactor;

  data->rs_k15[i] = data->r_k15[bin_id] +
    Tdef * (data->r_k15[bin_id+1] - data->r_k15[bin_id]);
  data->drs_k15[i] = (data->r_k15[bin_id+1] - data->r_k15[bin_id])*Tfactor;

  data->rs_k16[i] = data->r_k16[bin_id] +
    Tdef * (data->r_k16[bin_id+1] - data->r_k16[bin_id]);
  data->drs_k16[i] = (data->r_k16[bin_id+1] - data->r_k16[bin_id])*Tfactor;

  data->rs_k17[i] = data->r_k17[bin_id] +
    Tdef * (data->r_k17[bin_id+1] - data->r_k17[bin_id]);
  data->drs_k17[i] = (data->r_k17[bin_id+1] - data->r_k17[bin_id])*Tfactor;

  data->rs_k18[i] = data->r_k18[bin_id] +
    Tdef * (data->r_k18[bin_id+1] - data->r_k18[bin_id]);
  data->drs_k18[i] = (data->r_k18[bin_id+1] - data->r_k18[bin_id])*Tfactor;

  data->rs_k19[i] = data->r_k19[bin_id] +
    Tdef * (data->r_k19[bin_id+1] - data->r_k19[bin_id]);
  data->drs_k19[i] = (data->r_k19[bin_id+1] - data->r_k19[bin_id])*Tfactor;

  data->rs_k21[i] = data->r_k21[bin_id] +
    Tdef * (data->r_k21[bin_id+1] - data->r_k21[bin_id]);
  data->drs_k21[i] = (data->r_k21[bin_id+1] - data->r_k21[bin_id])*Tfactor;

  data->rs_k22[i] = data->r_k22[bin_id] +
    Tdef * (data->r_k22[bin_id+1] - data->r_k22[bin_id]);
  data->drs_k22[i] = (data->r_k22[bin_id+1] - data->r_k22[bin_id])*Tfactor;

  data->cs_brem_brem[i] = data->c_brem_brem[bin_id] +
    Tdef * (data->c_brem_brem[bin_id+1] - data->c_brem_brem[bin_id]);
  data->dcs_brem_brem[i] = (data->c_brem_brem[bin_id+1] - data->c_brem_brem[bin_id])*Tfactor;

  data->cs_ceHeI_ceHeI[i] = data->c_ceHeI_ceHeI[bin_id] +
    Tdef * (data->c_ceHeI_ceHeI[bin_id+1] - data->c_ceHeI_ceHeI[bin_id]);
  data->dcs_ceHeI_ceHeI[i] = (data->c_ceHeI_ceHeI[bin_id+1] - data->c_ceHeI_ceHeI[bin_id])*Tfactor;

  data->cs_ceHeII_ceHeII[i] = data->c_ceHeII_ceHeII[bin_id] +
    Tdef * (data->c_ceHeII_ceHeII[bin_id+1] - data->c_ceHeII_ceHeII[bin_id]);
  data->dcs_ceHeII_ceHeII[i] = (data->c_ceHeII_ceHeII[bin_id+1] - data->c_ceHeII_ceHeII[bin_id])*Tfactor;

  data->cs_ceHI_ceHI[i] = data->c_ceHI_ceHI[bin_id] +
    Tdef * (data->c_ceHI_ceHI[bin_id+1] - data->c_ceHI_ceHI[bin_id]);
  data->dcs_ceHI_ceHI[i] = (data->c_ceHI_ceHI[bin_id+1] - data->c_ceHI_ceHI[bin_id])*Tfactor;

  data->cs_cie_cooling_cieco[i] = data->c_cie_cooling_cieco[bin_id] +
    Tdef * (data->c_cie_cooling_cieco[bin_id+1] - data->c_cie_cooling_cieco[bin_id]);
  data->dcs_cie_cooling_cieco[i] = (data->c_cie_cooling_cieco[bin_id+1] - data->c_cie_cooling_cieco[bin_id])*Tfactor;

  data->cs_ciHeI_ciHeI[i] = data->c_ciHeI_ciHeI[bin_id] +
    Tdef * (data->c_ciHeI_ciHeI[bin_id+1] - data->c_ciHeI_ciHeI[bin_id]);
  data->dcs_ciHeI_ciHeI[i] = (data->c_ciHeI_ciHeI[bin_id+1] - data->c_ciHeI_ciHeI[bin_id])*Tfactor;

  data->cs_ciHeII_ciHeII[i] = data->c_ciHeII_ciHeII[bin_id] +
    Tdef * (data->c_ciHeII_ciHeII[bin_id+1] - data->c_ciHeII_ciHeII[bin_id]);
  data->dcs_ciHeII_ciHeII[i] = (data->c_ciHeII_ciHeII[bin_id+1] - data->c_ciHeII_ciHeII[bin_id])*Tfactor;

  data->cs_ciHeIS_ciHeIS[i] = data->c_ciHeIS_ciHeIS[bin_id] +
    Tdef * (data->c_ciHeIS_ciHeIS[bin_id+1] - data->c_ciHeIS_ciHeIS[bin_id]);
  data->dcs_ciHeIS_ciHeIS[i] = (data->c_ciHeIS_ciHeIS[bin_id+1] - data->c_ciHeIS_ciHeIS[bin_id])*Tfactor;

  data->cs_ciHI_ciHI[i] = data->c_ciHI_ciHI[bin_id] +
    Tdef * (data->c_ciHI_ciHI[bin_id+1] - data->c_ciHI_ciHI[bin_id]);
  data->dcs_ciHI_ciHI[i] = (data->c_ciHI_ciHI[bin_id+1] - data->c_ciHI_ciHI[bin_id])*Tfactor;

  data->cs_compton_comp_[i] = data->c_compton_comp_[bin_id] +
    Tdef * (data->c_compton_comp_[bin_id+1] - data->c_compton_comp_[bin_id]);
  data->dcs_compton_comp_[i] = (data->c_compton_comp_[bin_id+1] - data->c_compton_comp_[bin_id])*Tfactor;

  data->cs_gloverabel08_gael[i] = data->c_gloverabel08_gael[bin_id] +
    Tdef * (data->c_gloverabel08_gael[bin_id+1] - data->c_gloverabel08_gael[bin_id]);
  data->dcs_gloverabel08_gael[i] = (data->c_gloverabel08_gael[bin_id+1] - data->c_gloverabel08_gael[bin_id])*Tfactor;

  data->cs_gloverabel08_gaH2[i] = data->c_gloverabel08_gaH2[bin_id] +
    Tdef * (data->c_gloverabel08_gaH2[bin_id+1] - data->c_gloverabel08_gaH2[bin_id]);
  data->dcs_gloverabel08_gaH2[i] = (data->c_gloverabel08_gaH2[bin_id+1] - data->c_gloverabel08_gaH2[bin_id])*Tfactor;

  data->cs_gloverabel08_gaHe[i] = data->c_gloverabel08_gaHe[bin_id] +
    Tdef * (data->c_gloverabel08_gaHe[bin_id+1] - data->c_gloverabel08_gaHe[bin_id]);
  data->dcs_gloverabel08_gaHe[i] = (data->c_gloverabel08_gaHe[bin_id+1] - data->c_gloverabel08_gaHe[bin_id])*Tfactor;

  data->cs_gloverabel08_gaHI[i] = data->c_gloverabel08_gaHI[bin_id] +
    Tdef * (data->c_gloverabel08_gaHI[bin_id+1] - data->c_gloverabel08_gaHI[bin_id]);
  data->dcs_gloverabel08_gaHI[i] = (data->c_gloverabel08_gaHI[bin_id+1] - data->c_gloverabel08_gaHI[bin_id])*Tfactor;

  data->cs_gloverabel08_gaHp[i] = data->c_gloverabel08_gaHp[bin_id] +
    Tdef * (data->c_gloverabel08_gaHp[bin_id+1] - data->c_gloverabel08_gaHp[bin_id]);
  data->dcs_gloverabel08_gaHp[i] = (data->c_gloverabel08_gaHp[bin_id+1] - data->c_gloverabel08_gaHp[bin_id])*Tfactor;

  data->cs_gloverabel08_h2lte[i] = data->c_gloverabel08_h2lte[bin_id] +
    Tdef * (data->c_gloverabel08_h2lte[bin_id+1] - data->c_gloverabel08_h2lte[bin_id]);
  data->dcs_gloverabel08_h2lte[i] = (data->c_gloverabel08_h2lte[bin_id+1] - data->c_gloverabel08_h2lte[bin_id])*Tfactor;

  data->cs_h2formation_h2mcool[i] = data->c_h2formation_h2mcool[bin_id] +
    Tdef * (data->c_h2formation_h2mcool[bin_id+1] - data->c_h2formation_h2mcool[bin_id]);
  data->dcs_h2formation_h2mcool[i] = (data->c_h2formation_h2mcool[bin_id+1] - data->c_h2formation_h2mcool[bin_id])*Tfactor;

  data->cs_h2formation_h2mheat[i] = data->c_h2formation_h2mheat[bin_id] +
    Tdef * (data->c_h2formation_h2mheat[bin_id+1] - data->c_h2formation_h2mheat[bin_id]);
  data->dcs_h2formation_h2mheat[i] = (data->c_h2formation_h2mheat[bin_id+1] - data->c_h2formation_h2mheat[bin_id])*Tfactor;

  data->cs_h2formation_ncrd1[i] = data->c_h2formation_ncrd1[bin_id] +
    Tdef * (data->c_h2formation_ncrd1[bin_id+1] - data->c_h2formation_ncrd1[bin_id]);
  data->dcs_h2formation_ncrd1[i] = (data->c_h2formation_ncrd1[bin_id+1] - data->c_h2formation_ncrd1[bin_id])*Tfactor;

  data->cs_h2formation_ncrd2[i] = data->c_h2formation_ncrd2[bin_id] +
    Tdef * (data->c_h2formation_ncrd2[bin_id+1] - data->c_h2formation_ncrd2[bin_id]);
  data->dcs_h2formation_ncrd2[i] = (data->c_h2formation_ncrd2[bin_id+1] - data->c_h2formation_ncrd2[bin_id])*Tfactor;

  data->cs_h2formation_ncrn[i] = data->c_h2formation_ncrn[bin_id] +
    Tdef * (data->c_h2formation_ncrn[bin_id+1] - data->c_h2formation_ncrn[bin_id]);
  data->dcs_h2formation_ncrn[i] = (data->c_h2formation_ncrn[bin_id+1] - data->c_h2formation_ncrn[bin_id])*Tfactor;

  data->cs_reHeII1_reHeII1[i] = data->c_reHeII1_reHeII1[bin_id] +
    Tdef * (data->c_reHeII1_reHeII1[bin_id+1] - data->c_reHeII1_reHeII1[bin_id]);
  data->dcs_reHeII1_reHeII1[i] = (data->c_reHeII1_reHeII1[bin_id+1] - data->c_reHeII1_reHeII1[bin_id])*Tfactor;

  data->cs_reHeII2_reHeII2[i] = data->c_reHeII2_reHeII2[bin_id] +
    Tdef * (data->c_reHeII2_reHeII2[bin_id+1] - data->c_reHeII2_reHeII2[bin_id]);
  data->dcs_reHeII2_reHeII2[i] = (data->c_reHeII2_reHeII2[bin_id+1] - data->c_reHeII2_reHeII2[bin_id])*Tfactor;

  data->cs_reHeIII_reHeIII[i] = data->c_reHeIII_reHeIII[bin_id] +
    Tdef * (data->c_reHeIII_reHeIII[bin_id+1] - data->c_reHeIII_reHeIII[bin_id]);
  data->dcs_reHeIII_reHeIII[i] = (data->c_reHeIII_reHeIII[bin_id+1] - data->c_reHeIII_reHeIII[bin_id])*Tfactor;

  data->cs_reHII_reHII[i] = data->c_reHII_reHII[bin_id] +
    Tdef * (data->c_reHII_reHII[bin_id+1] - data->c_reHII_reHII[bin_id]);
  data->dcs_reHII_reHII[i] = (data->c_reHII_reHII[bin_id+1] - data->c_reHII_reHII[bin_id])*Tfactor;


}



RAJA_DEVICE void cvklu_interpolate_gamma(cvklu_data *data, long int i)
{

  int bin_id;
  double lb, t1, t2, Tdef;

  lb = log(data->bounds[0]);
  bin_id = (int) (data->idbin * (log(data->Ts[i]) - lb));
  if (bin_id <= 0) {
    bin_id = 0;
  } else if (bin_id >= data->nbins) {
    bin_id = data->nbins - 1;
  }
  t1 = (lb + (bin_id    ) * data->dbin);
  t2 = (lb + (bin_id + 1) * data->dbin);
  Tdef = (log(data->Ts[i]) - t1)/(t2 - t1);


  data->gammaH2_2[i] = data->g_gammaH2_2[bin_id] +
    Tdef * (data->g_gammaH2_2[bin_id+1] - data->g_gammaH2_2[bin_id]);

  data->dgammaH2_2_dT[i] = data->g_dgammaH2_2_dT[bin_id] +
    Tdef * (data->g_dgammaH2_2_dT[bin_id+1] - data->g_dgammaH2_2_dT[bin_id]);

  data->gammaH2_1[i] = data->g_gammaH2_1[bin_id] +
    Tdef * (data->g_gammaH2_1[bin_id+1] - data->g_gammaH2_1[bin_id]);

  data->dgammaH2_1_dT[i] = data->g_dgammaH2_1_dT[bin_id] +
    Tdef * (data->g_dgammaH2_1_dT[bin_id+1] - data->g_dgammaH2_1_dT[bin_id]);

}




int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  cvklu_data *data    = (cvklu_data*) user_data;
  double *scale       = data->scale;
  double *inv_scale   = data->inv_scale;
  const double *ydata = N_VGetDeviceArrayPointer(y);
  double *ydotdata    = N_VGetDeviceArrayPointer(ydot);

  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,data->nstrip), [=] RAJA_DEVICE (long int i) {

    double y_arr[NSPECIES];
    long int j = i * NSPECIES;
    double H2_1 = y_arr[0] = ydata[j]*scale[j];
    double H2_2 = y_arr[1] = ydata[j+1]*scale[j+1];
    double H_1 = y_arr[2]  = ydata[j+2]*scale[j+2];
    double H_2 = y_arr[3]  = ydata[j+3]*scale[j+3];
    double H_m0 = y_arr[4] = ydata[j+4]*scale[j+4];
    double He_1 = y_arr[5] = ydata[j+5]*scale[j+5];
    double He_2 = y_arr[6] = ydata[j+6]*scale[j+6];
    double He_3 = y_arr[7] = ydata[j+7]*scale[j+7];
    double de = y_arr[8]   = ydata[j+8]*scale[j+8];
    double ge = y_arr[9]   = ydata[j+9]*scale[j+9];

    // Calculate temperature in this cell
    cvklu_calculate_temperature(data, y_arr, i);
    // int flag = cvklu_calculate_temperature(data, y_arr, i);
    // if (flag > 0) {
    //   return 1;   // return recoverable failure if temperature failed to converged
    // }

    // Calculate reaction rates in this cell
    cvklu_interpolate_rates(data, i);

    // Set up some temporaries
    const double T = data->Ts[i];
    const double z = data->current_z;
    const double mdensity = data->mdensity[i];
    const double inv_mdensity = data->inv_mdensity[i];
    const double k01 = data->rs_k01[i];
    const double k02 = data->rs_k02[i];
    const double k03 = data->rs_k03[i];
    const double k04 = data->rs_k04[i];
    const double k05 = data->rs_k05[i];
    const double k06 = data->rs_k06[i];
    const double k07 = data->rs_k07[i];
    const double k08 = data->rs_k08[i];
    const double k09 = data->rs_k09[i];
    const double k10 = data->rs_k10[i];
    const double k11 = data->rs_k11[i];
    const double k12 = data->rs_k12[i];
    const double k13 = data->rs_k13[i];
    const double k14 = data->rs_k14[i];
    const double k15 = data->rs_k15[i];
    const double k16 = data->rs_k16[i];
    const double k17 = data->rs_k17[i];
    const double k18 = data->rs_k18[i];
    const double k19 = data->rs_k19[i];
    const double k21 = data->rs_k21[i];
    const double k22 = data->rs_k22[i];
    const double brem_brem = data->cs_brem_brem[i];
    const double ceHeI_ceHeI = data->cs_ceHeI_ceHeI[i];
    const double ceHeII_ceHeII = data->cs_ceHeII_ceHeII[i];
    const double ceHI_ceHI = data->cs_ceHI_ceHI[i];
    const double cie_cooling_cieco = data->cs_cie_cooling_cieco[i];
    const double ciHeI_ciHeI = data->cs_ciHeI_ciHeI[i];
    const double ciHeII_ciHeII = data->cs_ciHeII_ciHeII[i];
    const double ciHeIS_ciHeIS = data->cs_ciHeIS_ciHeIS[i];
    const double ciHI_ciHI = data->cs_ciHI_ciHI[i];
    const double compton_comp_ = data->cs_compton_comp_[i];
    const double gloverabel08_gael = data->cs_gloverabel08_gael[i];
    const double gloverabel08_gaH2 = data->cs_gloverabel08_gaH2[i];
    const double gloverabel08_gaHe = data->cs_gloverabel08_gaHe[i];
    const double gloverabel08_gaHI = data->cs_gloverabel08_gaHI[i];
    const double gloverabel08_gaHp = data->cs_gloverabel08_gaHp[i];
    const double gloverabel08_h2lte = data->cs_gloverabel08_h2lte[i];
    const double h2formation_h2mcool = data->cs_h2formation_h2mcool[i];
    const double h2formation_h2mheat = data->cs_h2formation_h2mheat[i];
    const double h2formation_ncrd1 = data->cs_h2formation_ncrd1[i];
    const double h2formation_ncrd2 = data->cs_h2formation_ncrd2[i];
    const double h2formation_ncrn = data->cs_h2formation_ncrn[i];
    const double reHeII1_reHeII1 = data->cs_reHeII1_reHeII1[i];
    const double reHeII2_reHeII2 = data->cs_reHeII2_reHeII2[i];
    const double reHeIII_reHeIII = data->cs_reHeIII_reHeIII[i];
    const double reHII_reHII = data->cs_reHII_reHII[i];
    const double h2_optical_depth_approx = data->h2_optical_depth_approx[i];
    const double cie_optical_depth_approx = data->cie_optical_depth_approx[i];

    j = i * NSPECIES;
    //
    // Species: H2_1
    //
    ydotdata[j] = k08*H_1*H_m0 + k10*H2_2*H_1 - k11*H2_1*H_2 - k12*H2_1*de - k13*H2_1*H_1 + k19*H2_2*H_m0 + k21*H2_1*H_1*H_1 + k22*H_1*H_1*H_1;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: H2_2
    //
    ydotdata[j] = k09*H_1*H_2 - k10*H2_2*H_1 + k11*H2_1*H_2 + k17*H_2*H_m0 - k18*H2_2*de - k19*H2_2*H_m0;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: H_1
    //
    ydotdata[j] = -k01*H_1*de + k02*H_2*de - k07*H_1*de - k08*H_1*H_m0 - k09*H_1*H_2 - k10*H2_2*H_1 + k11*H2_1*H_2 + 2*k12*H2_1*de + 2*k13*H2_1*H_1 + k14*H_m0*de + k15*H_1*H_m0 + 2*k16*H_2*H_m0 + 2*k18*H2_2*de + k19*H2_2*H_m0 - 2*k21*H2_1*H_1*H_1 - 2*k22*H_1*H_1*H_1;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: H_2
    //
    ydotdata[j] = k01*H_1*de - k02*H_2*de - k09*H_1*H_2 + k10*H2_2*H_1 - k11*H2_1*H_2 - k16*H_2*H_m0 - k17*H_2*H_m0;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: H_m0
    //
    ydotdata[j] = k07*H_1*de - k08*H_1*H_m0 - k14*H_m0*de - k15*H_1*H_m0 - k16*H_2*H_m0 - k17*H_2*H_m0 - k19*H2_2*H_m0;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: He_1
    //
    ydotdata[j] = -k03*He_1*de + k04*He_2*de;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: He_2
    //
    ydotdata[j] = k03*He_1*de - k04*He_2*de - k05*He_2*de + k06*He_3*de;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: He_3
    //
    ydotdata[j] = k05*He_2*de - k06*He_3*de;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: de
    //
    ydotdata[j] = k01*H_1*de - k02*H_2*de + k03*He_1*de - k04*He_2*de + k05*He_2*de - k06*He_3*de - k07*H_1*de + k08*H_1*H_m0 + k14*H_m0*de + k15*H_1*H_m0 + k17*H_2*H_m0 - k18*H2_2*de;
    ydotdata[j] *= inv_scale[j];
    j++;

    //
    // Species: ge
    //
    ydotdata[j] = -2.0158800000000001*H2_1*cie_cooling_cieco*cie_optical_depth_approx*mdensity - H2_1*cie_optical_depth_approx*gloverabel08_h2lte*h2_optical_depth_approx/(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0) - H_1*ceHI_ceHI*cie_optical_depth_approx*de - H_1*ciHI_ciHI*cie_optical_depth_approx*de - H_2*cie_optical_depth_approx*de*reHII_reHII - He_1*ciHeI_ciHeI*cie_optical_depth_approx*de - He_2*ceHeII_ceHeII*cie_optical_depth_approx*de - He_2*ceHeI_ceHeI*cie_optical_depth_approx*pow(de, 2) - He_2*ciHeII_ciHeII*cie_optical_depth_approx*de - He_2*ciHeIS_ciHeIS*cie_optical_depth_approx*pow(de, 2) - He_2*cie_optical_depth_approx*de*reHeII1_reHeII1 - He_2*cie_optical_depth_approx*de*reHeII2_reHeII2 - He_3*cie_optical_depth_approx*de*reHeIII_reHeIII - brem_brem*cie_optical_depth_approx*de*(H_2 + He_2 + 4.0*He_3) - cie_optical_depth_approx*compton_comp_*de*pow(z + 1.0, 4)*(T - 2.73*z - 2.73) + 0.5*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat);
    ydotdata[j] *= inv_scale[j];
    ydotdata[j] *= inv_mdensity;
    j++;

    //  }
  });

  return 0;
}




int calculate_sparse_jacobian_cvklu(realtype t, N_Vector y, N_Vector fy,
                                    SUNMatrix J, void *user_data,
                                    N_Vector tmp1, N_Vector tmp2,
                                    N_Vector tmp3)
{
  cvklu_data *data    = (cvklu_data*) user_data;
  const int NSPARSE   = 64;
  const double z      = data->current_z;
  double *scale       = data->scale;
  double *inv_scale   = data->inv_scale;
  const double *ydata = N_VGetDeviceArrayPointer(y);

  // Access CSR sparse matrix structures, and zero out data
  sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(J);
  sunindextype *colvals = SUNSparseMatrix_IndexValues(J);
  realtype *matrix_data = SUNSparseMatrix_Data(J);
  SUNMatZero(J);

  // Loop over data, filling in sparse Jacobian
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,data->nstrip), [=] RAJA_DEVICE (long int i) {

    // Set up some temporaries
    const double T   = data->Ts[i];
    const double Tge = data->dTs_ge[i];
    const double k01 = data->rs_k01[i];
    const double rk01= data->drs_k01[i];
    const double k02 = data->rs_k02[i];
    const double rk02= data->drs_k02[i];
    const double k03 = data->rs_k03[i];
    const double rk03= data->drs_k03[i];
    const double k04 = data->rs_k04[i];
    const double rk04= data->drs_k04[i];
    const double k05 = data->rs_k05[i];
    const double rk05= data->drs_k05[i];
    const double k06 = data->rs_k06[i];
    const double rk06= data->drs_k06[i];
    const double k07 = data->rs_k07[i];
    const double rk07= data->drs_k07[i];
    const double k08 = data->rs_k08[i];
    const double rk08= data->drs_k08[i];
    const double k09 = data->rs_k09[i];
    const double rk09= data->drs_k09[i];
    const double k10 = data->rs_k10[i];
    const double rk10= data->drs_k10[i];
    const double k11 = data->rs_k11[i];
    const double rk11= data->drs_k11[i];
    const double k12 = data->rs_k12[i];
    const double rk12= data->drs_k12[i];
    const double k13 = data->rs_k13[i];
    const double rk13= data->drs_k13[i];
    const double k14 = data->rs_k14[i];
    const double rk14= data->drs_k14[i];
    const double k15 = data->rs_k15[i];
    const double rk15= data->drs_k15[i];
    const double k16 = data->rs_k16[i];
    const double rk16= data->drs_k16[i];
    const double k17 = data->rs_k17[i];
    const double rk17= data->drs_k17[i];
    const double k18 = data->rs_k18[i];
    const double rk18= data->drs_k18[i];
    const double k19 = data->rs_k19[i];
    const double rk19= data->drs_k19[i];
    const double k21 = data->rs_k21[i];
    const double rk21= data->drs_k21[i];
    const double k22 = data->rs_k22[i];
    const double rk22= data->drs_k22[i];
    const double brem_brem = data->cs_brem_brem[i];
    const double ceHeI_ceHeI = data->cs_ceHeI_ceHeI[i];
    const double ceHeII_ceHeII = data->cs_ceHeII_ceHeII[i];
    const double ceHI_ceHI = data->cs_ceHI_ceHI[i];
    const double cie_cooling_cieco = data->cs_cie_cooling_cieco[i];
    const double ciHeI_ciHeI = data->cs_ciHeI_ciHeI[i];
    const double ciHeII_ciHeII = data->cs_ciHeII_ciHeII[i];
    const double ciHeIS_ciHeIS = data->cs_ciHeIS_ciHeIS[i];
    const double ciHI_ciHI = data->cs_ciHI_ciHI[i];
    const double compton_comp_ = data->cs_compton_comp_[i];
    const double gloverabel08_gael = data->cs_gloverabel08_gael[i];
    const double rgloverabel08_gael = data->dcs_gloverabel08_gael[i];
    const double gloverabel08_gaH2 = data->cs_gloverabel08_gaH2[i];
    const double rgloverabel08_gaH2 = data->dcs_gloverabel08_gaH2[i];
    const double gloverabel08_gaHe = data->cs_gloverabel08_gaHe[i];
    const double rgloverabel08_gaHe = data->dcs_gloverabel08_gaHe[i];
    const double gloverabel08_gaHI = data->cs_gloverabel08_gaHI[i];
    const double rgloverabel08_gaHI = data->dcs_gloverabel08_gaHI[i];
    const double gloverabel08_gaHp = data->cs_gloverabel08_gaHp[i];
    const double rgloverabel08_gaHp = data->dcs_gloverabel08_gaHp[i];
    const double gloverabel08_h2lte = data->cs_gloverabel08_h2lte[i];
    const double rgloverabel08_h2lte = data->dcs_gloverabel08_h2lte[i];
    const double h2formation_h2mcool = data->cs_h2formation_h2mcool[i];
    const double rh2formation_h2mcool = data->dcs_h2formation_h2mcool[i];
    const double h2formation_h2mheat = data->cs_h2formation_h2mheat[i];
    const double rh2formation_h2mheat = data->dcs_h2formation_h2mheat[i];
    const double h2formation_ncrd1 = data->cs_h2formation_ncrd1[i];
    const double rh2formation_ncrd1 = data->dcs_h2formation_ncrd1[i];
    const double h2formation_ncrd2 = data->cs_h2formation_ncrd2[i];
    const double rh2formation_ncrd2 = data->dcs_h2formation_ncrd2[i];
    const double h2formation_ncrn = data->cs_h2formation_ncrn[i];
    const double rh2formation_ncrn = data->dcs_h2formation_ncrn[i];
    const double reHeII1_reHeII1 = data->cs_reHeII1_reHeII1[i];
    const double reHeII2_reHeII2 = data->cs_reHeII2_reHeII2[i];
    const double reHeIII_reHeIII = data->cs_reHeIII_reHeIII[i];
    const double reHII_reHII = data->cs_reHII_reHII[i];

    long int j = i * NSPECIES;
    const double H2_1 = ydata[j]*scale[j];
    const double H2_2 = ydata[j+1]*scale[j+1];
    const double H_1  = ydata[j+2]*scale[j+2];
    const double H_2  = ydata[j+3]*scale[j+3];
    const double H_m0 = ydata[j+4]*scale[j+4];
    const double He_1 = ydata[j+5]*scale[j+5];
    const double He_2 = ydata[j+6]*scale[j+6];
    const double He_3 = ydata[j+7]*scale[j+7];
    const double de   = ydata[j+8]*scale[j+8];
    const double ge   = ydata[j+9]*scale[j+9];
    const double mdensity     = data->mdensity[i];
    const double inv_mdensity = 1.0 / mdensity;
    const double h2_optical_depth_approx  = data->h2_optical_depth_approx[i];
    const double cie_optical_depth_approx = data->cie_optical_depth_approx[i];

    j = i * NSPARSE;

    // H2_1 by H2_1
    colvals[j + 0] = i * NSPECIES + 0 ;
    matrix_data[ j + 0 ] = -k11*H_2 - k12*de - k13*H_1 + k21*pow(H_1, 2);

    // H2_1 by H2_2
    colvals[j + 1] = i * NSPECIES + 1 ;
    matrix_data[ j + 1 ] = k10*H_1 + k19*H_m0;

    // H2_1 by H_1
    colvals[j + 2] = i * NSPECIES + 2 ;
    matrix_data[ j + 2 ] = k08*H_m0 + k10*H2_2 - k13*H2_1 + 2*k21*H2_1*H_1 + 3*k22*pow(H_1, 2);

    // H2_1 by H_2
    colvals[j + 3] = i * NSPECIES + 3 ;
    matrix_data[ j + 3 ] = -k11*H2_1;

    // H2_1 by H_m0
    colvals[j + 4] = i * NSPECIES + 4 ;
    matrix_data[ j + 4 ] = k08*H_1 + k19*H2_2;

    // H2_1 by de
    colvals[j + 5] = i * NSPECIES + 8 ;
    matrix_data[ j + 5 ] = -k12*H2_1;

    // H2_1 by ge
    colvals[j + 6] = i * NSPECIES + 9 ;
    matrix_data[ j + 6 ] = rk08*H_1*H_m0 + rk10*H2_2*H_1 - rk11*H2_1*H_2 - rk12*H2_1*de - rk13*H2_1*H_1 + rk19*H2_2*H_m0 + rk21*H2_1*H_1*H_1 + rk22*H_1*H_1*H_1;
    matrix_data[ j + 6] *= Tge;

    // H2_2 by H2_1
    colvals[j + 7] = i * NSPECIES + 0 ;
    matrix_data[ j + 7 ] = k11*H_2;

    // H2_2 by H2_2
    colvals[j + 8] = i * NSPECIES + 1 ;
    matrix_data[ j + 8 ] = -k10*H_1 - k18*de - k19*H_m0;

    // H2_2 by H_1
    colvals[j + 9] = i * NSPECIES + 2 ;
    matrix_data[ j + 9 ] = k09*H_2 - k10*H2_2;

    // H2_2 by H_2
    colvals[j + 10] = i * NSPECIES + 3 ;
    matrix_data[ j + 10 ] = k09*H_1 + k11*H2_1 + k17*H_m0;

    // H2_2 by H_m0
    colvals[j + 11] = i * NSPECIES + 4 ;
    matrix_data[ j + 11 ] = k17*H_2 - k19*H2_2;

    // H2_2 by de
    colvals[j + 12] = i * NSPECIES + 8 ;
    matrix_data[ j + 12 ] = -k18*H2_2;

    // H2_2 by ge
    colvals[j + 13] = i * NSPECIES + 9 ;
    matrix_data[ j + 13 ] = rk09*H_1*H_2 - rk10*H2_2*H_1 + rk11*H2_1*H_2 + rk17*H_2*H_m0 - rk18*H2_2*de - rk19*H2_2*H_m0;
    matrix_data[ j + 13] *= Tge;

    // H_1 by H2_1
    colvals[j + 14] = i * NSPECIES + 0 ;
    matrix_data[ j + 14 ] = k11*H_2 + 2*k12*de + 2*k13*H_1 - 2*k21*pow(H_1, 2);

    // H_1 by H2_2
    colvals[j + 15] = i * NSPECIES + 1 ;
    matrix_data[ j + 15 ] = -k10*H_1 + 2*k18*de + k19*H_m0;

    // H_1 by H_1
    colvals[j + 16] = i * NSPECIES + 2 ;
    matrix_data[ j + 16 ] = -k01*de - k07*de - k08*H_m0 - k09*H_2 - k10*H2_2 + 2*k13*H2_1 + k15*H_m0 - 4*k21*H2_1*H_1 - 6*k22*pow(H_1, 2);

    // H_1 by H_2
    colvals[j + 17] = i * NSPECIES + 3 ;
    matrix_data[ j + 17 ] = k02*de - k09*H_1 + k11*H2_1 + 2*k16*H_m0;

    // H_1 by H_m0
    colvals[j + 18] = i * NSPECIES + 4 ;
    matrix_data[ j + 18 ] = -k08*H_1 + k14*de + k15*H_1 + 2*k16*H_2 + k19*H2_2;

    // H_1 by de
    colvals[j + 19] = i * NSPECIES + 8 ;
    matrix_data[ j + 19 ] = -k01*H_1 + k02*H_2 - k07*H_1 + 2*k12*H2_1 + k14*H_m0 + 2*k18*H2_2;

    // H_1 by ge
    colvals[j + 20] = i * NSPECIES + 9 ;
    matrix_data[ j + 20 ] = -rk01*H_1*de + rk02*H_2*de - rk07*H_1*de - rk08*H_1*H_m0 - rk09*H_1*H_2 - rk10*H2_2*H_1 + rk11*H2_1*H_2 + 2*rk12*H2_1*de + 2*rk13*H2_1*H_1 + rk14*H_m0*de + rk15*H_1*H_m0 + 2*rk16*H_2*H_m0 + 2*rk18*H2_2*de + rk19*H2_2*H_m0 - 2*rk21*H2_1*H_1*H_1 - 2*rk22*H_1*H_1*H_1;
    matrix_data[ j + 20] *= Tge;

    // H_2 by H2_1
    colvals[j + 21] = i * NSPECIES + 0 ;
    matrix_data[ j + 21 ] = -k11*H_2;

    // H_2 by H2_2
    colvals[j + 22] = i * NSPECIES + 1 ;
    matrix_data[ j + 22 ] = k10*H_1;

    // H_2 by H_1
    colvals[j + 23] = i * NSPECIES + 2 ;
    matrix_data[ j + 23 ] = k01*de - k09*H_2 + k10*H2_2;

    // H_2 by H_2
    colvals[j + 24] = i * NSPECIES + 3 ;
    matrix_data[ j + 24 ] = -k02*de - k09*H_1 - k11*H2_1 - k16*H_m0 - k17*H_m0;

    // H_2 by H_m0
    colvals[j + 25] = i * NSPECIES + 4 ;
    matrix_data[ j + 25 ] = -k16*H_2 - k17*H_2;

    // H_2 by de
    colvals[j + 26] = i * NSPECIES + 8 ;
    matrix_data[ j + 26 ] = k01*H_1 - k02*H_2;

    // H_2 by ge
    colvals[j + 27] = i * NSPECIES + 9 ;
    matrix_data[ j + 27 ] = rk01*H_1*de - rk02*H_2*de - rk09*H_1*H_2 + rk10*H2_2*H_1 - rk11*H2_1*H_2 - rk16*H_2*H_m0 - rk17*H_2*H_m0;
    matrix_data[ j + 27] *= Tge;

    // H_m0 by H2_2
    colvals[j + 28] = i * NSPECIES + 1 ;
    matrix_data[ j + 28 ] = -k19*H_m0;

    // H_m0 by H_1
    colvals[j + 29] = i * NSPECIES + 2 ;
    matrix_data[ j + 29 ] = k07*de - k08*H_m0 - k15*H_m0;

    // H_m0 by H_2
    colvals[j + 30] = i * NSPECIES + 3 ;
    matrix_data[ j + 30 ] = -k16*H_m0 - k17*H_m0;

    // H_m0 by H_m0
    colvals[j + 31] = i * NSPECIES + 4 ;
    matrix_data[ j + 31 ] = -k08*H_1 - k14*de - k15*H_1 - k16*H_2 - k17*H_2 - k19*H2_2;

    // H_m0 by de
    colvals[j + 32] = i * NSPECIES + 8 ;
    matrix_data[ j + 32 ] = k07*H_1 - k14*H_m0;

    // H_m0 by ge
    colvals[j + 33] = i * NSPECIES + 9 ;
    matrix_data[ j + 33 ] = rk07*H_1*de - rk08*H_1*H_m0 - rk14*H_m0*de - rk15*H_1*H_m0 - rk16*H_2*H_m0 - rk17*H_2*H_m0 - rk19*H2_2*H_m0;
    matrix_data[ j + 33] *= Tge;

    // He_1 by He_1
    colvals[j + 34] = i * NSPECIES + 5 ;
    matrix_data[ j + 34 ] = -k03*de;

    // He_1 by He_2
    colvals[j + 35] = i * NSPECIES + 6 ;
    matrix_data[ j + 35 ] = k04*de;

    // He_1 by de
    colvals[j + 36] = i * NSPECIES + 8 ;
    matrix_data[ j + 36 ] = -k03*He_1 + k04*He_2;

    // He_1 by ge
    colvals[j + 37] = i * NSPECIES + 9 ;
    matrix_data[ j + 37 ] = -rk03*He_1*de + rk04*He_2*de;
    matrix_data[ j + 37] *= Tge;

    // He_2 by He_1
    colvals[j + 38] = i * NSPECIES + 5 ;
    matrix_data[ j + 38 ] = k03*de;

    // He_2 by He_2
    colvals[j + 39] = i * NSPECIES + 6 ;
    matrix_data[ j + 39 ] = -k04*de - k05*de;

    // He_2 by He_3
    colvals[j + 40] = i * NSPECIES + 7 ;
    matrix_data[ j + 40 ] = k06*de;

    // He_2 by de
    colvals[j + 41] = i * NSPECIES + 8 ;
    matrix_data[ j + 41 ] = k03*He_1 - k04*He_2 - k05*He_2 + k06*He_3;

    // He_2 by ge
    colvals[j + 42] = i * NSPECIES + 9 ;
    matrix_data[ j + 42 ] = rk03*He_1*de - rk04*He_2*de - rk05*He_2*de + rk06*He_3*de;
    matrix_data[ j + 42] *= Tge;

    // He_3 by He_2
    colvals[j + 43] = i * NSPECIES + 6 ;
    matrix_data[ j + 43 ] = k05*de;

    // He_3 by He_3
    colvals[j + 44] = i * NSPECIES + 7 ;
    matrix_data[ j + 44 ] = -k06*de;

    // He_3 by de
    colvals[j + 45] = i * NSPECIES + 8 ;
    matrix_data[ j + 45 ] = k05*He_2 - k06*He_3;

    // He_3 by ge
    colvals[j + 46] = i * NSPECIES + 9 ;
    matrix_data[ j + 46 ] = rk05*He_2*de - rk06*He_3*de;
    matrix_data[ j + 46] *= Tge;

    // de by H2_2
    colvals[j + 47] = i * NSPECIES + 1 ;
    matrix_data[ j + 47 ] = -k18*de;

    // de by H_1
    colvals[j + 48] = i * NSPECIES + 2 ;
    matrix_data[ j + 48 ] = k01*de - k07*de + k08*H_m0 + k15*H_m0;

    // de by H_2
    colvals[j + 49] = i * NSPECIES + 3 ;
    matrix_data[ j + 49 ] = -k02*de + k17*H_m0;

    // de by H_m0
    colvals[j + 50] = i * NSPECIES + 4 ;
    matrix_data[ j + 50 ] = k08*H_1 + k14*de + k15*H_1 + k17*H_2;

    // de by He_1
    colvals[j + 51] = i * NSPECIES + 5 ;
    matrix_data[ j + 51 ] = k03*de;

    // de by He_2
    colvals[j + 52] = i * NSPECIES + 6 ;
    matrix_data[ j + 52 ] = -k04*de + k05*de;

    // de by He_3
    colvals[j + 53] = i * NSPECIES + 7 ;
    matrix_data[ j + 53 ] = -k06*de;

    // de by de
    colvals[j + 54] = i * NSPECIES + 8 ;
    matrix_data[ j + 54 ] = k01*H_1 - k02*H_2 + k03*He_1 - k04*He_2 + k05*He_2 - k06*He_3 - k07*H_1 + k14*H_m0 - k18*H2_2;

    // de by ge
    colvals[j + 55] = i * NSPECIES + 9 ;
    matrix_data[ j + 55 ] = rk01*H_1*de - rk02*H_2*de + rk03*He_1*de - rk04*He_2*de + rk05*He_2*de - rk06*He_3*de - rk07*H_1*de + rk08*H_1*H_m0 + rk14*H_m0*de + rk15*H_1*H_m0 + rk17*H_2*H_m0 - rk18*H2_2*de;
    matrix_data[ j + 55] *= Tge;

    // ge by H2_1
    colvals[j + 56] = i * NSPECIES + 0 ;
    matrix_data[ j + 56 ] = -H2_1*gloverabel08_gaH2*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - 0.5*H_1*h2formation_h2mcool*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0) - 2.0158800000000001*cie_cooling_cieco*mdensity - gloverabel08_h2lte*h2_optical_depth_approx/(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0) + 0.5*h2formation_ncrd2*h2formation_ncrn*pow(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat)/pow(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1, 2);
    matrix_data[j + 56] *= inv_mdensity;

    // ge by H_1
    colvals[j + 57] = i * NSPECIES + 2 ;
    matrix_data[ j + 57 ] = -H2_1*gloverabel08_gaHI*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - ceHI_ceHI*de - ciHI_ciHI*de + 0.5*h2formation_ncrd1*h2formation_ncrn*pow(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat)/pow(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1, 2) + 0.5*(-H2_1*h2formation_h2mcool + 3*pow(H_1, 2)*h2formation_h2mheat)*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0);
    matrix_data[j + 57] *= inv_mdensity;

    // ge by H_2
    colvals[j + 58] = i * NSPECIES + 3 ;
    matrix_data[ j + 58 ] = -H2_1*gloverabel08_gaHp*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - brem_brem*de - de*reHII_reHII;
    matrix_data[j + 58] *= inv_mdensity;

    // ge by He_1
    colvals[j + 59] = i * NSPECIES + 5 ;
    matrix_data[ j + 59 ] = -H2_1*gloverabel08_gaHe*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - ciHeI_ciHeI*de;
    matrix_data[j + 59] *= inv_mdensity;

    // ge by He_2
    colvals[j + 60] = i * NSPECIES + 6 ;
    matrix_data[ j + 60 ] = -brem_brem*de - ceHeII_ceHeII*de - ceHeI_ceHeI*pow(de, 2) - ciHeII_ciHeII*de - ciHeIS_ciHeIS*pow(de, 2) - de*reHeII1_reHeII1 - de*reHeII2_reHeII2;
    matrix_data[j + 60] *= inv_mdensity;

    // ge by He_3
    colvals[j + 61] = i * NSPECIES + 7 ;
    matrix_data[ j + 61 ] = -4.0*brem_brem*de - de*reHeIII_reHeIII;
    matrix_data[j + 61] *= inv_mdensity;

    // ge by de
    colvals[j + 62] = i * NSPECIES + 8 ;
    matrix_data[ j + 62 ] = -H2_1*gloverabel08_gael*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - H_1*ceHI_ceHI - H_1*ciHI_ciHI - H_2*reHII_reHII - He_1*ciHeI_ciHeI - He_2*ceHeII_ceHeII - 2*He_2*ceHeI_ceHeI*de - He_2*ciHeII_ciHeII - 2*He_2*ciHeIS_ciHeIS*de - He_2*reHeII1_reHeII1 - He_2*reHeII2_reHeII2 - He_3*reHeIII_reHeIII - brem_brem*(H_2 + He_2 + 4.0*He_3) - compton_comp_*pow(z + 1.0, 4)*(T - 2.73*z - 2.73);
    matrix_data[j + 62] *= inv_mdensity;

    // ge by ge
    colvals[j + 63] = i * NSPECIES + 9 ;
    matrix_data[ j + 63 ] = -2.0158800000000001*H2_1*cie_cooling_cieco*cie_optical_depth_approx*mdensity - H2_1*cie_optical_depth_approx*gloverabel08_h2lte*h2_optical_depth_approx/(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0) - H_1*ceHI_ceHI*cie_optical_depth_approx*de - H_1*ciHI_ciHI*cie_optical_depth_approx*de - H_2*cie_optical_depth_approx*de*reHII_reHII - He_1*ciHeI_ciHeI*cie_optical_depth_approx*de - He_2*ceHeII_ceHeII*cie_optical_depth_approx*de - He_2*ceHeI_ceHeI*cie_optical_depth_approx*pow(de, 2) - He_2*ciHeII_ciHeII*cie_optical_depth_approx*de - He_2*ciHeIS_ciHeIS*cie_optical_depth_approx*pow(de, 2) - He_2*cie_optical_depth_approx*de*reHeII1_reHeII1 - He_2*cie_optical_depth_approx*de*reHeII2_reHeII2 - He_3*cie_optical_depth_approx*de*reHeIII_reHeIII - brem_brem*cie_optical_depth_approx*de*(H_2 + He_2 + 4.0*He_3) - cie_optical_depth_approx*compton_comp_*de*pow(z + 1.0, 4)*(T - 2.73*z - 2.73) + 0.5*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat);

    // ad-hoc extra term of f_ge by ge
    // considering ONLY the h2formation/ and continuum cooling
    matrix_data[ j + 63] = -H2_1*gloverabel08_h2lte*h2_optical_depth_approx*(-gloverabel08_h2lte*(-H2_1*rgloverabel08_gaH2 - H_1*rgloverabel08_gaHI - H_2*rgloverabel08_gaHp - He_1*rgloverabel08_gaHe - de*rgloverabel08_gael)/pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2) - rgloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael))/pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2) - H2_1*h2_optical_depth_approx*rgloverabel08_h2lte/(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0) + 0.5*pow(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat)*(-1.0*h2formation_ncrn*(-H2_1*rh2formation_ncrd2 - H_1*rh2formation_ncrd1)/pow(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1, 2) - 1.0*rh2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1)) + 0.5*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0)*(-H2_1*H_1*rh2formation_h2mcool + pow(H_1, 3)*rh2formation_h2mheat);
    matrix_data[ j + 63] *= inv_mdensity;
    matrix_data[ j + 63] *= Tge;

    rowptrs[ i * NSPECIES +  0] = i * NSPARSE + 0;
    rowptrs[ i * NSPECIES +  1] = i * NSPARSE + 7;
    rowptrs[ i * NSPECIES +  2] = i * NSPARSE + 14;
    rowptrs[ i * NSPECIES +  3] = i * NSPARSE + 21;
    rowptrs[ i * NSPECIES +  4] = i * NSPARSE + 28;
    rowptrs[ i * NSPECIES +  5] = i * NSPARSE + 34;
    rowptrs[ i * NSPECIES +  6] = i * NSPARSE + 38;
    rowptrs[ i * NSPECIES +  7] = i * NSPARSE + 43;
    rowptrs[ i * NSPECIES +  8] = i * NSPARSE + 47;
    rowptrs[ i * NSPECIES +  9] = i * NSPARSE + 56;

    j = i * NSPECIES;
    matrix_data[ i * NSPARSE + 0]  *=  (inv_scale[ j + 0 ]*scale[ j + 0 ]);
    matrix_data[ i * NSPARSE + 1]  *=  (inv_scale[ j + 0 ]*scale[ j + 1 ]);
    matrix_data[ i * NSPARSE + 2]  *=  (inv_scale[ j + 0 ]*scale[ j + 2 ]);
    matrix_data[ i * NSPARSE + 3]  *= (inv_scale[ j + 0 ]*scale[ j + 3 ]);
    matrix_data[ i * NSPARSE + 4]  *= (inv_scale[ j + 0 ]*scale[ j + 4 ]);
    matrix_data[ i * NSPARSE + 5]  *= (inv_scale[ j + 0 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 6]  *= (inv_scale[ j + 0 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 7]  *= (inv_scale[ j + 1 ]*scale[ j + 0 ]);
    matrix_data[ i * NSPARSE + 8]  *= (inv_scale[ j + 1 ]*scale[ j + 1 ]);
    matrix_data[ i * NSPARSE + 9]  *= (inv_scale[ j + 1 ]*scale[ j + 2 ]);
    matrix_data[ i * NSPARSE + 10]  *= (inv_scale[ j + 1 ]*scale[ j + 3 ]);
    matrix_data[ i * NSPARSE + 11]  *= (inv_scale[ j + 1 ]*scale[ j + 4 ]);
    matrix_data[ i * NSPARSE + 12]  *= (inv_scale[ j + 1 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 13]  *= (inv_scale[ j + 1 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 14]  *= (inv_scale[ j + 2 ]*scale[ j + 0 ]);
    matrix_data[ i * NSPARSE + 15]  *= (inv_scale[ j + 2 ]*scale[ j + 1 ]);
    matrix_data[ i * NSPARSE + 16]  *= (inv_scale[ j + 2 ]*scale[ j + 2 ]);
    matrix_data[ i * NSPARSE + 17]  *= (inv_scale[ j + 2 ]*scale[ j + 3 ]);
    matrix_data[ i * NSPARSE + 18]  *= (inv_scale[ j + 2 ]*scale[ j + 4 ]);
    matrix_data[ i * NSPARSE + 19]  *= (inv_scale[ j + 2 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 20]  *= (inv_scale[ j + 2 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 21]  *= (inv_scale[ j + 3 ]*scale[ j + 0 ]);
    matrix_data[ i * NSPARSE + 22]  *= (inv_scale[ j + 3 ]*scale[ j + 1 ]);
    matrix_data[ i * NSPARSE + 23]  *= (inv_scale[ j + 3 ]*scale[ j + 2 ]);
    matrix_data[ i * NSPARSE + 24]  *= (inv_scale[ j + 3 ]*scale[ j + 3 ]);
    matrix_data[ i * NSPARSE + 25]  *= (inv_scale[ j + 3 ]*scale[ j + 4 ]);
    matrix_data[ i * NSPARSE + 26]  *= (inv_scale[ j + 3 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 27]  *= (inv_scale[ j + 3 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 28]  *= (inv_scale[ j + 4 ]*scale[ j + 1 ]);
    matrix_data[ i * NSPARSE + 29]  *= (inv_scale[ j + 4 ]*scale[ j + 2 ]);
    matrix_data[ i * NSPARSE + 30]  *= (inv_scale[ j + 4 ]*scale[ j + 3 ]);
    matrix_data[ i * NSPARSE + 31]  *= (inv_scale[ j + 4 ]*scale[ j + 4 ]);
    matrix_data[ i * NSPARSE + 32]  *= (inv_scale[ j + 4 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 33]  *= (inv_scale[ j + 4 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 34]  *= (inv_scale[ j + 5 ]*scale[ j + 5 ]);
    matrix_data[ i * NSPARSE + 35]  *= (inv_scale[ j + 5 ]*scale[ j + 6 ]);
    matrix_data[ i * NSPARSE + 36]  *= (inv_scale[ j + 5 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 37]  *= (inv_scale[ j + 5 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 38]  *= (inv_scale[ j + 6 ]*scale[ j + 5 ]);
    matrix_data[ i * NSPARSE + 39]  *= (inv_scale[ j + 6 ]*scale[ j + 6 ]);
    matrix_data[ i * NSPARSE + 40]  *= (inv_scale[ j + 6 ]*scale[ j + 7 ]);
    matrix_data[ i * NSPARSE + 41]  *= (inv_scale[ j + 6 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 42]  *= (inv_scale[ j + 6 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 43]  *= (inv_scale[ j + 7 ]*scale[ j + 6 ]);
    matrix_data[ i * NSPARSE + 44]  *= (inv_scale[ j + 7 ]*scale[ j + 7 ]);
    matrix_data[ i * NSPARSE + 45]  *= (inv_scale[ j + 7 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 46]  *= (inv_scale[ j + 7 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 47]  *= (inv_scale[ j + 8 ]*scale[ j + 1 ]);
    matrix_data[ i * NSPARSE + 48]  *= (inv_scale[ j + 8 ]*scale[ j + 2 ]);
    matrix_data[ i * NSPARSE + 49]  *= (inv_scale[ j + 8 ]*scale[ j + 3 ]);
    matrix_data[ i * NSPARSE + 50]  *= (inv_scale[ j + 8 ]*scale[ j + 4 ]);
    matrix_data[ i * NSPARSE + 51]  *= (inv_scale[ j + 8 ]*scale[ j + 5 ]);
    matrix_data[ i * NSPARSE + 52]  *= (inv_scale[ j + 8 ]*scale[ j + 6 ]);
    matrix_data[ i * NSPARSE + 53]  *= (inv_scale[ j + 8 ]*scale[ j + 7 ]);
    matrix_data[ i * NSPARSE + 54]  *= (inv_scale[ j + 8 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 55]  *= (inv_scale[ j + 8 ]*scale[ j + 9 ]);
    matrix_data[ i * NSPARSE + 56]  *= (inv_scale[ j + 9 ]*scale[ j + 0 ]);
    matrix_data[ i * NSPARSE + 57]  *= (inv_scale[ j + 9 ]*scale[ j + 2 ]);
    matrix_data[ i * NSPARSE + 58]  *= (inv_scale[ j + 9 ]*scale[ j + 3 ]);
    matrix_data[ i * NSPARSE + 59]  *= (inv_scale[ j + 9 ]*scale[ j + 5 ]);
    matrix_data[ i * NSPARSE + 60]  *= (inv_scale[ j + 9 ]*scale[ j + 6 ]);
    matrix_data[ i * NSPARSE + 61]  *= (inv_scale[ j + 9 ]*scale[ j + 7 ]);
    matrix_data[ i * NSPARSE + 62]  *= (inv_scale[ j + 9 ]*scale[ j + 8 ]);
    matrix_data[ i * NSPARSE + 63]  *= (inv_scale[ j + 9 ]*scale[ j + 9 ]);

  });

  rowptrs[ data->nstrip * NSPECIES ] = data->nstrip * NSPARSE;
  return 0;
}




void setting_up_extra_variables( cvklu_data * data, double * input, long int nstrip ){

  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,nstrip), [=] RAJA_DEVICE (long int i) {

    double mdensity = 0.0;

    // species: H2_1
    mdensity += input[i * NSPECIES] * 2.0;

    // species: H2_2
    mdensity += input[i * NSPECIES + 1] * 2.0;

    // species: H_1
    mdensity += input[i * NSPECIES + 2] * 1.00794;

    // species: H_2
    mdensity += input[i * NSPECIES + 3] * 1.00794;

    // species: H_m0
    mdensity += input[i * NSPECIES + 4] * 1.00794;

    // species: He_1
    mdensity += input[i * NSPECIES + 5] * 4.002602;

    // species: He_2
    mdensity += input[i * NSPECIES + 6] * 4.002602;

    // species: He_3
    mdensity += input[i * NSPECIES + 7] * 4.002602;

    // scale mdensity by Hydrogen mass
    mdensity *= 1.67e-24;

    double tau = pow( (mdensity / 3.3e-8 ), 2.8);
    tau = fmax( tau, 1.0e-5 );

    // store results
    data->mdensity[i] = mdensity;
    data->inv_mdensity[i] = 1.0 / mdensity;
    data->cie_optical_depth_approx[i] = fmin( 1.0, (1.0 - exp(-tau) ) / tau );
    data->h2_optical_depth_approx[i] = fmin( 1.0, pow( (mdensity / (1.34e-14) )  , -0.45) );
  });



}
