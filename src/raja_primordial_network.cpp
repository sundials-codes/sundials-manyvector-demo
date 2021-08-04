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

// HIP vs RAJA vs serial
#if defined(RAJA_CUDA)
#define HIP_OR_CUDA(a,b) b
#define dcopy(dst, src, len) cudaMemcpy(dst, src, (len)*sizeof(double), cudaMemcpyHostToDevice)
#elif defined(RAJA_HIP)
#define HIP_OR_CUDA(a,b) a
#define dcopy(dst, src, len) hipMemcpy(dst, src, (len)*sizeof(double), hipMemcpyHostToDevice)
#else
#define HIP_OR_CUDA(a,b) ((void)0);
#define dcopy(dst, src, len) memcpy(dst, src, (len)*sizeof(double))
#endif

// sparse vs dense
#define NSPARSE 64
#if defined(USEMAGMA)
#define SPARSE_OR_DENSE(a,b) b
#else
#define SPARSE_OR_DENSE(a,b) a
#endif
#define SPARSEIDX(blk,off) (blk*NSPARSE + off)
#define DENSEIDX(blk,row,col) (blk*NSPECIES*NSPECIES + col*NSPECIES + row)

cvklu_data *cvklu_setup_data(const char *FileLocation, long int ncells, SUNMemoryHelper memhelper)
{

  //-----------------------------------------------------
  // Function : cvklu_setup_data
  // Description: Initialize a data object that stores the reaction/ cooling rate data
  //-----------------------------------------------------

  cvklu_data *data = NULL;
#ifdef RAJA_SERIAL
  data = (cvklu_data *) malloc(sizeof(cvklu_data));
#elif RAJA_CUDA
  cudaMallocManaged((void**)&(data), sizeof(cvklu_data));
#else
  hipMalloc((void**)&(data), sizeof(cvklu_data));
#endif

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
  double *Ts_dev = data->Ts;
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,ncells), [=] RAJA_DEVICE (long int i) {
    Ts_dev[i] = 1000.0;
  });

  // Temperature-related pieces
  data->bounds[0] = 1.0;
  data->bounds[1] = 100000.0;
  data->nbins = 1024 - 1;
  data->dbin = (log(data->bounds[1]) - log(data->bounds[0])) / data->nbins;
  data->idbin = 1.0L / data->dbin;

#ifdef RAJA_SERIAL
  data->r_k01 = (double *) malloc(1024*sizeof(double));
  data->r_k02 = (double *) malloc(1024*sizeof(double));
  data->r_k03 = (double *) malloc(1024*sizeof(double));
  data->r_k04 = (double *) malloc(1024*sizeof(double));
  data->r_k05 = (double *) malloc(1024*sizeof(double));
  data->r_k06 = (double *) malloc(1024*sizeof(double));
  data->r_k07 = (double *) malloc(1024*sizeof(double));
  data->r_k08 = (double *) malloc(1024*sizeof(double));
  data->r_k09 = (double *) malloc(1024*sizeof(double));
  data->r_k10 = (double *) malloc(1024*sizeof(double));
  data->r_k11 = (double *) malloc(1024*sizeof(double));
  data->r_k12 = (double *) malloc(1024*sizeof(double));
  data->r_k13 = (double *) malloc(1024*sizeof(double));
  data->r_k14 = (double *) malloc(1024*sizeof(double));
  data->r_k15 = (double *) malloc(1024*sizeof(double));
  data->r_k16 = (double *) malloc(1024*sizeof(double));
  data->r_k17 = (double *) malloc(1024*sizeof(double));
  data->r_k18 = (double *) malloc(1024*sizeof(double));
  data->r_k19 = (double *) malloc(1024*sizeof(double));
  data->r_k21 = (double *) malloc(1024*sizeof(double));
  data->r_k22 = (double *) malloc(1024*sizeof(double));
  data->c_brem_brem = (double *) malloc(1024*sizeof(double));
  data->c_ceHeI_ceHeI = (double *) malloc(1024*sizeof(double));
  data->c_ceHeII_ceHeII = (double *) malloc(1024*sizeof(double));
  data->c_ceHI_ceHI = (double *) malloc(1024*sizeof(double));
  data->c_cie_cooling_cieco = (double *) malloc(1024*sizeof(double));
  data->c_ciHeI_ciHeI = (double *) malloc(1024*sizeof(double));
  data->c_ciHeII_ciHeII = (double *) malloc(1024*sizeof(double));
  data->c_ciHeIS_ciHeIS = (double *) malloc(1024*sizeof(double));
  data->c_ciHI_ciHI = (double *) malloc(1024*sizeof(double));
  data->c_compton_comp_ = (double *) malloc(1024*sizeof(double));
  data->c_gloverabel08_gael = (double *) malloc(1024*sizeof(double));
  data->c_gloverabel08_gaH2 = (double *) malloc(1024*sizeof(double));
  data->c_gloverabel08_gaHe = (double *) malloc(1024*sizeof(double));
  data->c_gloverabel08_gaHI = (double *) malloc(1024*sizeof(double));
  data->c_gloverabel08_gaHp = (double *) malloc(1024*sizeof(double));
  data->c_gloverabel08_h2lte = (double *) malloc(1024*sizeof(double));
  data->c_h2formation_h2mcool = (double *) malloc(1024*sizeof(double));
  data->c_h2formation_h2mheat = (double *) malloc(1024*sizeof(double));
  data->c_h2formation_ncrd1 = (double *) malloc(1024*sizeof(double));
  data->c_h2formation_ncrd2 = (double *) malloc(1024*sizeof(double));
  data->c_h2formation_ncrn = (double *) malloc(1024*sizeof(double));
  data->c_reHeII1_reHeII1 = (double *) malloc(1024*sizeof(double));
  data->c_reHeII2_reHeII2 = (double *) malloc(1024*sizeof(double));
  data->c_reHeIII_reHeIII = (double *) malloc(1024*sizeof(double));
  data->c_reHII_reHII = (double *) malloc(1024*sizeof(double));
  data->g_gammaH2_1 = (double *) malloc(1024*sizeof(double));
  data->g_dgammaH2_1_dT = (double *) malloc(1024*sizeof(double));
  data->g_gammaH2_2 = (double *) malloc(1024*sizeof(double));
  data->g_dgammaH2_2_dT = (double *) malloc(1024*sizeof(double));
#elif RAJA_CUDA
  cudaMallocManaged((void**)&(data->r_k01), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k02), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k03), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k04), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k05), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k06), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k07), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k08), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k09), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k10), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k11), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k12), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k13), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k14), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k15), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k16), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k17), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k18), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k19), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k21), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->r_k22), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_brem_brem), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_ceHeI_ceHeI), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_ceHeII_ceHeII), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_ceHI_ceHI), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_cie_cooling_cieco), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_ciHeI_ciHeI), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_ciHeII_ciHeII), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_ciHeIS_ciHeIS), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_ciHI_ciHI), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_compton_comp_), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_gloverabel08_gael), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_gloverabel08_gaH2), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_gloverabel08_gaHe), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_gloverabel08_gaHI), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_gloverabel08_gaHp), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_gloverabel08_h2lte), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_h2formation_h2mcool), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_h2formation_h2mheat), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_h2formation_ncrd1), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_h2formation_ncrd2), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_h2formation_ncrn), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_reHeII1_reHeII1), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_reHeII2_reHeII2), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_reHeIII_reHeIII), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->c_reHII_reHII), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->g_gammaH2_1), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->g_dgammaH2_1_dT), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->g_gammaH2_2), 1024*sizeof(double));
  cudaMallocManaged((void**)&(data->g_dgammaH2_2_dT), 1024*sizeof(double));
#else
#error RAJA HIP chemistry interface is currently unimplemented
#endif
  cvklu_read_rate_tables(data);
  cvklu_read_cooling_tables(data);
  cvklu_read_gamma(data);
  return data;
}


void cvklu_free_data(void *data, SUNMemoryHelper memhelper)
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
  free(rxdata->cie_optical_depth_approx);
  free(rxdata->h2_optical_depth_approx);
  free(rxdata->r_k01);
  free(rxdata->r_k02);
  free(rxdata->r_k03);
  free(rxdata->r_k04);
  free(rxdata->r_k05);
  free(rxdata->r_k06);
  free(rxdata->r_k07);
  free(rxdata->r_k08);
  free(rxdata->r_k09);
  free(rxdata->r_k10);
  free(rxdata->r_k11);
  free(rxdata->r_k12);
  free(rxdata->r_k13);
  free(rxdata->r_k14);
  free(rxdata->r_k15);
  free(rxdata->r_k16);
  free(rxdata->r_k17);
  free(rxdata->r_k18);
  free(rxdata->r_k19);
  free(rxdata->r_k21);
  free(rxdata->r_k22);
  free(rxdata->c_brem_brem);
  free(rxdata->c_ceHeI_ceHeI);
  free(rxdata->c_ceHeII_ceHeII);
  free(rxdata->c_ceHI_ceHI);
  free(rxdata->c_cie_cooling_cieco);
  free(rxdata->c_ciHeI_ciHeI);
  free(rxdata->c_ciHeII_ciHeII);
  free(rxdata->c_ciHeIS_ciHeIS);
  free(rxdata->c_ciHI_ciHI);
  free(rxdata->c_compton_comp_);
  free(rxdata->c_gloverabel08_gael);
  free(rxdata->c_gloverabel08_gaH2);
  free(rxdata->c_gloverabel08_gaHe);
  free(rxdata->c_gloverabel08_gaHI);
  free(rxdata->c_gloverabel08_gaHp);
  free(rxdata->c_gloverabel08_h2lte);
  free(rxdata->c_h2formation_h2mcool);
  free(rxdata->c_h2formation_h2mheat);
  free(rxdata->c_h2formation_ncrd1);
  free(rxdata->c_h2formation_ncrd2);
  free(rxdata->c_h2formation_ncrn);
  free(rxdata->c_reHeII1_reHeII1);
  free(rxdata->c_reHeII2_reHeII2);
  free(rxdata->c_reHeIII_reHeIII);
  free(rxdata->c_reHII_reHII);
  free(rxdata->g_gammaH2_1);
  free(rxdata->g_dgammaH2_1_dT);
  free(rxdata->g_gammaH2_2);
  free(rxdata->g_dgammaH2_2_dT);
  free(rxdata);
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
  cudaFree(rxdata->cie_optical_depth_approx);
  cudaFree(rxdata->h2_optical_depth_approx);
  cudaFree(rxdata->r_k01);
  cudaFree(rxdata->r_k02);
  cudaFree(rxdata->r_k03);
  cudaFree(rxdata->r_k04);
  cudaFree(rxdata->r_k05);
  cudaFree(rxdata->r_k06);
  cudaFree(rxdata->r_k07);
  cudaFree(rxdata->r_k08);
  cudaFree(rxdata->r_k09);
  cudaFree(rxdata->r_k10);
  cudaFree(rxdata->r_k11);
  cudaFree(rxdata->r_k12);
  cudaFree(rxdata->r_k13);
  cudaFree(rxdata->r_k14);
  cudaFree(rxdata->r_k15);
  cudaFree(rxdata->r_k16);
  cudaFree(rxdata->r_k17);
  cudaFree(rxdata->r_k18);
  cudaFree(rxdata->r_k19);
  cudaFree(rxdata->r_k21);
  cudaFree(rxdata->r_k22);
  cudaFree(rxdata->c_brem_brem);
  cudaFree(rxdata->c_ceHeI_ceHeI);
  cudaFree(rxdata->c_ceHeII_ceHeII);
  cudaFree(rxdata->c_ceHI_ceHI);
  cudaFree(rxdata->c_cie_cooling_cieco);
  cudaFree(rxdata->c_ciHeI_ciHeI);
  cudaFree(rxdata->c_ciHeII_ciHeII);
  cudaFree(rxdata->c_ciHeIS_ciHeIS);
  cudaFree(rxdata->c_ciHI_ciHI);
  cudaFree(rxdata->c_compton_comp_);
  cudaFree(rxdata->c_gloverabel08_gael);
  cudaFree(rxdata->c_gloverabel08_gaH2);
  cudaFree(rxdata->c_gloverabel08_gaHe);
  cudaFree(rxdata->c_gloverabel08_gaHI);
  cudaFree(rxdata->c_gloverabel08_gaHp);
  cudaFree(rxdata->c_gloverabel08_h2lte);
  cudaFree(rxdata->c_h2formation_h2mcool);
  cudaFree(rxdata->c_h2formation_h2mheat);
  cudaFree(rxdata->c_h2formation_ncrd1);
  cudaFree(rxdata->c_h2formation_ncrd2);
  cudaFree(rxdata->c_h2formation_ncrn);
  cudaFree(rxdata->c_reHeII1_reHeII1);
  cudaFree(rxdata->c_reHeII2_reHeII2);
  cudaFree(rxdata->c_reHeIII_reHeIII);
  cudaFree(rxdata->c_reHII_reHII);
  cudaFree(rxdata->g_gammaH2_1);
  cudaFree(rxdata->g_dgammaH2_1_dT);
  cudaFree(rxdata->g_gammaH2_2);
  cudaFree(rxdata->g_dgammaH2_2_dT);
  cudaFree(rxdata);
#else
#error RAJA HIP chemistry interface is currently unimplemented
#endif

}


// UPDATE THIS TO RETURN SUCCESS/FAILURE FLAG
void cvklu_read_rate_tables(cvklu_data *data)
{
  // Allocate temporary memory on host for file input
  double *k01 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k02 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k03 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k04 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k05 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k06 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k07 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k08 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k09 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k10 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k11 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k12 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k13 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k14 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k15 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k16 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k17 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k18 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k19 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k21 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *k22 = (double*) malloc((data->nbins+1)*sizeof(double));

  // Read the rate tables to temporaries
  const char * filedir;
  if (data->dengo_data_file != NULL){
    filedir =  data->dengo_data_file;
  } else{
    filedir = "cvklu_tables.h5";
  }
  hid_t file_id = H5Fopen( filedir , H5F_ACC_RDONLY, H5P_DEFAULT);
  H5LTread_dataset_double(file_id, "/k01", k01);
  H5LTread_dataset_double(file_id, "/k02", k02);
  H5LTread_dataset_double(file_id, "/k03", k03);
  H5LTread_dataset_double(file_id, "/k04", k04);
  H5LTread_dataset_double(file_id, "/k05", k05);
  H5LTread_dataset_double(file_id, "/k06", k06);
  H5LTread_dataset_double(file_id, "/k07", k07);
  H5LTread_dataset_double(file_id, "/k08", k08);
  H5LTread_dataset_double(file_id, "/k09", k09);
  H5LTread_dataset_double(file_id, "/k10", k10);
  H5LTread_dataset_double(file_id, "/k11", k11);
  H5LTread_dataset_double(file_id, "/k12", k12);
  H5LTread_dataset_double(file_id, "/k13", k13);
  H5LTread_dataset_double(file_id, "/k14", k14);
  H5LTread_dataset_double(file_id, "/k15", k15);
  H5LTread_dataset_double(file_id, "/k16", k16);
  H5LTread_dataset_double(file_id, "/k17", k17);
  H5LTread_dataset_double(file_id, "/k18", k18);
  H5LTread_dataset_double(file_id, "/k19", k19);
  H5LTread_dataset_double(file_id, "/k21", k21);
  H5LTread_dataset_double(file_id, "/k22", k22);
  H5Fclose(file_id);

  // Copy tables into rate data structure
  dcopy(data->r_k01, k01, data->nbins+1);
  dcopy(data->r_k02, k02, data->nbins+1);
  dcopy(data->r_k03, k03, data->nbins+1);
  dcopy(data->r_k04, k04, data->nbins+1);
  dcopy(data->r_k05, k05, data->nbins+1);
  dcopy(data->r_k06, k06, data->nbins+1);
  dcopy(data->r_k07, k07, data->nbins+1);
  dcopy(data->r_k08, k08, data->nbins+1);
  dcopy(data->r_k09, k09, data->nbins+1);
  dcopy(data->r_k10, k10, data->nbins+1);
  dcopy(data->r_k11, k11, data->nbins+1);
  dcopy(data->r_k12, k12, data->nbins+1);
  dcopy(data->r_k13, k13, data->nbins+1);
  dcopy(data->r_k14, k14, data->nbins+1);
  dcopy(data->r_k15, k15, data->nbins+1);
  dcopy(data->r_k16, k16, data->nbins+1);
  dcopy(data->r_k17, k17, data->nbins+1);
  dcopy(data->r_k18, k18, data->nbins+1);
  dcopy(data->r_k19, k19, data->nbins+1);
  dcopy(data->r_k21, k21, data->nbins+1);
  dcopy(data->r_k22, k22, data->nbins+1);

  // ensure that table data is synchronized between host/device memory
  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )
  HIP_OR_CUDA( hipError_t cuerr = hipGetLastError();,
               cudaError_t cuerr = cudaGetLastError(); )
#if defined(RAJA_CUDA) || defined(RAJA_HIP)
  if (cuerr != HIP_OR_CUDA( hipSuccess, cudaSuccess )) {
      std::cerr << ">>> ERROR in cvklu_read_rate_tables: XGetLastError returned %s\n"
              << HIP_OR_CUDA( hipGetErrorName(cuerr), cudaGetErrorName(cuerr) );
    return;
  }
#endif

  // Free temporary arrays
  free(k01);
  free(k02);
  free(k03);
  free(k04);
  free(k05);
  free(k06);
  free(k07);
  free(k08);
  free(k09);
  free(k10);
  free(k11);
  free(k12);
  free(k13);
  free(k14);
  free(k15);
  free(k16);
  free(k17);
  free(k18);
  free(k19);
  free(k21);
  free(k22);
}


// UPDATE THIS TO RETURN SUCCESS/FAILURE FLAG
void cvklu_read_cooling_tables(cvklu_data *data)
{
  // Allocate temporary memory on host for file input
  double *c_brem_brem = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_ceHeI_ceHeI = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_ceHeII_ceHeII = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_ceHI_ceHI = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_cie_cooling_cieco = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_ciHeI_ciHeI = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_ciHeII_ciHeII = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_ciHeIS_ciHeIS = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_ciHI_ciHI = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_compton_comp_ = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_gloverabel08_gael = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_gloverabel08_gaH2 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_gloverabel08_gaHe = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_gloverabel08_gaHI = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_gloverabel08_gaHp = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_gloverabel08_h2lte = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_h2formation_h2mcool = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_h2formation_h2mheat = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_h2formation_ncrd1 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_h2formation_ncrd2 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_h2formation_ncrn = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_reHeII1_reHeII1 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_reHeII2_reHeII2 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_reHeIII_reHeIII = (double*) malloc((data->nbins+1)*sizeof(double));
  double *c_reHII_reHII = (double*) malloc((data->nbins+1)*sizeof(double));

  // Read the cooling tables to temporaries
  const char * filedir;
  if (data->dengo_data_file != NULL){
    filedir =  data->dengo_data_file;
  } else{
    filedir = "cvklu_tables.h5";
  }
  hid_t file_id = H5Fopen( filedir , H5F_ACC_RDONLY, H5P_DEFAULT);
  H5LTread_dataset_double(file_id, "/brem_brem",           c_brem_brem);
  H5LTread_dataset_double(file_id, "/ceHeI_ceHeI",         c_ceHeI_ceHeI);
  H5LTread_dataset_double(file_id, "/ceHeII_ceHeII",       c_ceHeII_ceHeII);
  H5LTread_dataset_double(file_id, "/ceHI_ceHI",           c_ceHI_ceHI);
  H5LTread_dataset_double(file_id, "/cie_cooling_cieco",   c_cie_cooling_cieco);
  H5LTread_dataset_double(file_id, "/ciHeI_ciHeI",         c_ciHeI_ciHeI);
  H5LTread_dataset_double(file_id, "/ciHeII_ciHeII",       c_ciHeII_ciHeII);
  H5LTread_dataset_double(file_id, "/ciHeIS_ciHeIS",       c_ciHeIS_ciHeIS);
  H5LTread_dataset_double(file_id, "/ciHI_ciHI",           c_ciHI_ciHI);
  H5LTread_dataset_double(file_id, "/compton_comp_",       c_compton_comp_);
  H5LTread_dataset_double(file_id, "/gloverabel08_gael",   c_gloverabel08_gael);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaH2",   c_gloverabel08_gaH2);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHe",   c_gloverabel08_gaHe);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHI",   c_gloverabel08_gaHI);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHp",   c_gloverabel08_gaHp);
  H5LTread_dataset_double(file_id, "/gloverabel08_h2lte",  c_gloverabel08_h2lte);
  H5LTread_dataset_double(file_id, "/h2formation_h2mcool", c_h2formation_h2mcool);
  H5LTread_dataset_double(file_id, "/h2formation_h2mheat", c_h2formation_h2mheat);
  H5LTread_dataset_double(file_id, "/h2formation_ncrd1",   c_h2formation_ncrd1);
  H5LTread_dataset_double(file_id, "/h2formation_ncrd2",   c_h2formation_ncrd2);
  H5LTread_dataset_double(file_id, "/h2formation_ncrn",    c_h2formation_ncrn);
  H5LTread_dataset_double(file_id, "/reHeII1_reHeII1",     c_reHeII1_reHeII1);
  H5LTread_dataset_double(file_id, "/reHeII2_reHeII2",     c_reHeII2_reHeII2);
  H5LTread_dataset_double(file_id, "/reHeIII_reHeIII",     c_reHeIII_reHeIII);
  H5LTread_dataset_double(file_id, "/reHII_reHII",         c_reHII_reHII);
  H5Fclose(file_id);

  // Copy tables into rate data structure
  dcopy(data->c_brem_brem, c_brem_brem, data->nbins+1);
  dcopy(data->c_ceHeI_ceHeI, c_ceHeI_ceHeI, data->nbins+1);
  dcopy(data->c_ceHeII_ceHeII, c_ceHeII_ceHeII, data->nbins+1);
  dcopy(data->c_ceHI_ceHI, c_ceHI_ceHI, data->nbins+1);
  dcopy(data->c_cie_cooling_cieco, c_cie_cooling_cieco, data->nbins+1);
  dcopy(data->c_ciHeI_ciHeI, c_ciHeI_ciHeI, data->nbins+1);
  dcopy(data->c_ciHeII_ciHeII, c_ciHeII_ciHeII, data->nbins+1);
  dcopy(data->c_ciHeIS_ciHeIS, c_ciHeIS_ciHeIS, data->nbins+1);
  dcopy(data->c_ciHI_ciHI, c_ciHI_ciHI, data->nbins+1);
  dcopy(data->c_compton_comp_, c_compton_comp_, data->nbins+1);
  dcopy(data->c_gloverabel08_gael, c_gloverabel08_gael, data->nbins+1);
  dcopy(data->c_gloverabel08_gaH2, c_gloverabel08_gaH2, data->nbins+1);
  dcopy(data->c_gloverabel08_gaHe, c_gloverabel08_gaHe, data->nbins+1);
  dcopy(data->c_gloverabel08_gaHI, c_gloverabel08_gaHI, data->nbins+1);
  dcopy(data->c_gloverabel08_gaHp, c_gloverabel08_gaHp, data->nbins+1);
  dcopy(data->c_gloverabel08_h2lte, c_gloverabel08_h2lte, data->nbins+1);
  dcopy(data->c_h2formation_h2mcool, c_h2formation_h2mcool, data->nbins+1);
  dcopy(data->c_h2formation_h2mheat, c_h2formation_h2mheat, data->nbins+1);
  dcopy(data->c_h2formation_ncrd1, c_h2formation_ncrd1, data->nbins+1);
  dcopy(data->c_h2formation_ncrd2, c_h2formation_ncrd2, data->nbins+1);
  dcopy(data->c_h2formation_ncrn, c_h2formation_ncrn, data->nbins+1);
  dcopy(data->c_reHeII1_reHeII1, c_reHeII1_reHeII1, data->nbins+1);
  dcopy(data->c_reHeII2_reHeII2, c_reHeII2_reHeII2, data->nbins+1);
  dcopy(data->c_reHeIII_reHeIII, c_reHeIII_reHeIII, data->nbins+1);
  dcopy(data->c_reHII_reHII, c_reHII_reHII, data->nbins+1);

  // ensure that table data is synchronized between host/device memory
  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )
  HIP_OR_CUDA( hipError_t cuerr = hipGetLastError();,
               cudaError_t cuerr = cudaGetLastError(); )
#if defined(RAJA_CUDA) || defined(RAJA_HIP)
  if (cuerr != HIP_OR_CUDA( hipSuccess, cudaSuccess )) {
      std::cerr << ">>> ERROR in cvklu_read_rate_tables: XGetLastError returned %s\n"
              << HIP_OR_CUDA( hipGetErrorName(cuerr), cudaGetErrorName(cuerr) );
    return;
  }
#endif

  // Free temporary arrays
  free(c_brem_brem);
  free(c_ceHeI_ceHeI);
  free(c_ceHeII_ceHeII);
  free(c_ceHI_ceHI);
  free(c_cie_cooling_cieco);
  free(c_ciHeI_ciHeI);
  free(c_ciHeII_ciHeII);
  free(c_ciHeIS_ciHeIS);
  free(c_ciHI_ciHI);
  free(c_compton_comp_);
  free(c_gloverabel08_gael);
  free(c_gloverabel08_gaH2);
  free(c_gloverabel08_gaHe);
  free(c_gloverabel08_gaHI);
  free(c_gloverabel08_gaHp);
  free(c_gloverabel08_h2lte);
  free(c_h2formation_h2mcool);
  free(c_h2formation_h2mheat);
  free(c_h2formation_ncrd1);
  free(c_h2formation_ncrd2);
  free(c_h2formation_ncrn);
  free(c_reHeII1_reHeII1);
  free(c_reHeII2_reHeII2);
  free(c_reHeIII_reHeIII);
  free(c_reHII_reHII);
}

// UPDATE THIS TO RETURN SUCCESS/FAILURE FLAG
void cvklu_read_gamma(cvklu_data *data)
{

  // Allocate temporary memory on host for file input
  double *g_gammaH2_1 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *g_dgammaH2_1_dT = (double*) malloc((data->nbins+1)*sizeof(double));
  double *g_gammaH2_2 = (double*) malloc((data->nbins+1)*sizeof(double));
  double *g_dgammaH2_2_dT = (double*) malloc((data->nbins+1)*sizeof(double));

  // Read the gamma tables to temporaries
  const char * filedir;
  if (data->dengo_data_file != NULL){
    filedir =  data->dengo_data_file;
  } else{
    filedir = "cvklu_tables.h5";
  }
  hid_t file_id = H5Fopen( filedir , H5F_ACC_RDONLY, H5P_DEFAULT);
  H5LTread_dataset_double(file_id, "/gammaH2_1",     data->g_gammaH2_1 );
  H5LTread_dataset_double(file_id, "/dgammaH2_1_dT", data->g_dgammaH2_1_dT );
  H5LTread_dataset_double(file_id, "/gammaH2_2",     data->g_gammaH2_2 );
  H5LTread_dataset_double(file_id, "/dgammaH2_2_dT", data->g_dgammaH2_2_dT );
  H5Fclose(file_id);

  // Copy tables into rate data structure
  dcopy(data->g_gammaH2_1, g_gammaH2_1, data->nbins+1);
  dcopy(data->g_dgammaH2_1_dT, g_dgammaH2_1_dT, data->nbins+1);
  dcopy(data->g_gammaH2_2, g_gammaH2_2, data->nbins+1);
  dcopy(data->g_dgammaH2_2_dT, g_dgammaH2_2_dT, data->nbins+1);

  // ensure that table data is synchronized between host/device memory
  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )
  HIP_OR_CUDA( hipError_t cuerr = hipGetLastError();,
               cudaError_t cuerr = cudaGetLastError(); )
#if defined(RAJA_CUDA) || defined(RAJA_HIP)
  if (cuerr != HIP_OR_CUDA( hipSuccess, cudaSuccess )) {
      std::cerr << ">>> ERROR in cvklu_read_rate_tables: XGetLastError returned %s\n"
              << HIP_OR_CUDA( hipGetErrorName(cuerr), cudaGetErrorName(cuerr) );
    return;
  }
#endif

  // Free temporary arrays
  free(g_gammaH2_1);
  free(g_dgammaH2_1_dT);
  free(g_gammaH2_2);
  free(g_dgammaH2_2_dT);
}



RAJA_DEVICE int cvklu_calculate_temperature(const cvklu_data *data, const double *y_arr,
                                            const long int i, double &Ts, double &dTs_ge)
{

  // Define some constants
  const double kb = 1.3806504e-16; // Boltzmann constant [erg/K]
  const double mh = 1.67e-24;
  const double gamma = 5.e0/3.e0;
  const double _gamma_m1 = 1.0 / (gamma - 1);
  //const int MAX_T_ITERATION = 100;

  // Calculate total density
  double H2_1 = y_arr[0];
  double H2_2 = y_arr[1];
  double H_1 = y_arr[2];
  double H_2 = y_arr[3];
  double H_m0 = y_arr[4];
  double He_1 = y_arr[5];
  double He_2 = y_arr[6];
  double He_3 = y_arr[7];
  double de = y_arr[8];
  double ge = y_arr[9];
  double density = 2.0*H2_1 + 2.0*H2_2 + 1.0079400000000001*H_1
    + 1.0079400000000001*H_2 + 1.0079400000000001*H_m0 + 4.0026020000000004*He_1
    + 4.0026020000000004*He_2 + 4.0026020000000004*He_3;

  // Initiate the "guess" temperature
  double T    = Ts;
  double Tnew = T*1.1;
  double Tdiff = Tnew - T;
  double dge_dT;
  //int count = 0;

  // We do Newton's Iteration to calculate the temperature
  // Since gammaH2 is dependent on the temperature too!
  //        while ( Tdiff/ Tnew > 0.001 ){
  for (int j=0; j<10; j++){

    T = Ts;

    // interpolate gamma
    int bin_id;
    double lb, t1, t2, Tdef;

    lb = log(data->bounds[0]);
    bin_id = (int) (data->idbin * (log(Ts) - lb));
    if (bin_id <= 0) {
      bin_id = 0;
    } else if (bin_id >= data->nbins) {
      bin_id = data->nbins - 1;
    }
    t1 = (lb + (bin_id    ) * data->dbin);
    t2 = (lb + (bin_id + 1) * data->dbin);
    Tdef = (log(Ts) - t1)/(t2 - t1);

    double gammaH2_2 = data->g_gammaH2_2[bin_id] +
      Tdef * (data->g_gammaH2_2[bin_id+1] - data->g_gammaH2_2[bin_id]);

    double dgammaH2_2_dT = data->g_dgammaH2_2_dT[bin_id] +
      Tdef * (data->g_dgammaH2_2_dT[bin_id+1] - data->g_dgammaH2_2_dT[bin_id]);

    double gammaH2_1 = data->g_gammaH2_1[bin_id] +
      Tdef * (data->g_gammaH2_1[bin_id+1] - data->g_gammaH2_1[bin_id]);

    double dgammaH2_1_dT = data->g_dgammaH2_1_dT[bin_id] +
      Tdef * (data->g_dgammaH2_1_dT[bin_id+1] - data->g_dgammaH2_1_dT[bin_id]);
    ////

    double _gammaH2_1_m1 = 1.0 / (gammaH2_1 - 1.0);
    double _gammaH2_2_m1 = 1.0 / (gammaH2_2 - 1.0);

    // update gammaH2
    // The derivatives of  sum (nkT/(gamma - 1)/mh/density) - ge
    // This is the function we want to minimize
    // which should only be dependent on the first part
    dge_dT = T*kb*(-H2_1*_gammaH2_1_m1*_gammaH2_1_m1*dgammaH2_1_dT - H2_2*_gammaH2_2_m1*_gammaH2_2_m1*dgammaH2_2_dT)/(density*mh) + kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh);

    //This is the change in ge for each iteration
    double dge = T*kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh) - ge;

    Tnew = T - dge/dge_dT;
    Ts = Tnew;

    Tdiff = fabs(T - Tnew);
    // fprintf(stderr, "T: %0.5g ; Tnew: %0.5g; dge_dT: %.5g, dge: %.5g, ge: %.5g \n", T,Tnew, dge_dT, dge, ge);
    // count += 1;
    // if (count > MAX_T_ITERATION){
    //     fprintf(stderr, "T failed to converge \n");
    //     return 1;
    // }
  } // while loop

  Ts = Tnew;

  if (Ts < data->bounds[0]) {
    Ts = data->bounds[0];
  } else if (Ts > data->bounds[1]) {
    Ts = data->bounds[1];
  }
  dTs_ge = 1.0 / dge_dT;

  return 0;

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
    const double H2_1 = y_arr[0] = ydata[j]*scale[j];
    const double H2_2 = y_arr[1] = ydata[j+1]*scale[j+1];
    const double H_1 = y_arr[2]  = ydata[j+2]*scale[j+2];
    const double H_2 = y_arr[3]  = ydata[j+3]*scale[j+3];
    const double H_m0 = y_arr[4] = ydata[j+4]*scale[j+4];
    const double He_1 = y_arr[5] = ydata[j+5]*scale[j+5];
    const double He_2 = y_arr[6] = ydata[j+6]*scale[j+6];
    const double He_3 = y_arr[7] = ydata[j+7]*scale[j+7];
    const double de = y_arr[8]   = ydata[j+8]*scale[j+8];
    const double ge = y_arr[9]   = ydata[j+9]*scale[j+9];

    // Calculate temperature in this cell
    cvklu_calculate_temperature(data, y_arr, i, data->Ts[i], data->dTs_ge[i]);

    // Calculate reaction rates in this cell
    //cvklu_interpolate_rates(data, i);
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
    ////

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

  });

  // synchronize device memory
  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )
  HIP_OR_CUDA( hipError_t cuerr = hipGetLastError();,
               cudaError_t cuerr = cudaGetLastError(); )
  if (cuerr != HIP_OR_CUDA( hipSuccess, cudaSuccess )) {
    std::cerr << ">>> ERROR in calculate_rhs_cvklu: XGetLastError returned %s\n"
              << HIP_OR_CUDA( hipGetErrorName(cuerr), cudaGetErrorName(cuerr) );
    return(-1);
  }

  return 0;
}



#ifndef USEMAGMA
int initialize_sparse_jacobian_cvklu( SUNMatrix J, void *user_data )
{
#ifdef RAJA_CUDA

  // Access CSR sparse matrix structures, and zero out data
  sunindextype rowptrs[11];
  sunindextype colvals[NSPARSE];

  // H2_1 by H2_1
  colvals[0] = 0 ;

  // H2_1 by H2_2
  colvals[1] = 1 ;

  // H2_1 by H_1
  colvals[2] = 2 ;

  // H2_1 by H_2
  colvals[3] = 3 ;

  // H2_1 by H_m0
  colvals[4] = 4 ;

  // H2_1 by de
  colvals[5] = 8 ;

  // H2_1 by ge
  colvals[6] = 9 ;

  // H2_2 by H2_1
  colvals[7] = 0 ;

  // H2_2 by H2_2
  colvals[8] = 1 ;

  // H2_2 by H_1
  colvals[9] = 2 ;

  // H2_2 by H_2
  colvals[10] = 3 ;

  // H2_2 by H_m0
  colvals[11] = 4 ;

  // H2_2 by de
  colvals[12] = 8 ;

  // H2_2 by ge
  colvals[13] = 9 ;

  // H_1 by H2_1
  colvals[14] = 0 ;

  // H_1 by H2_2
  colvals[15] = 1 ;

  // H_1 by H_1
  colvals[16] = 2 ;

  // H_1 by H_2
  colvals[17] = 3 ;

  // H_1 by H_m0
  colvals[18] = 4 ;

  // H_1 by de
  colvals[19] = 8 ;

  // H_1 by ge
  colvals[20] = 9 ;

  // H_2 by H2_1
  colvals[21] = 0 ;

  // H_2 by H2_2
  colvals[22] = 1 ;

  // H_2 by H_1
  colvals[23] = 2 ;

  // H_2 by H_2
  colvals[24] = 3 ;

  // H_2 by H_m0
  colvals[25] = 4 ;

  // H_2 by de
  colvals[26] = 8 ;

  // H_2 by ge
  colvals[27] = 9 ;

  // H_m0 by H2_2
  colvals[28] = 1 ;

  // H_m0 by H_1
  colvals[29] = 2 ;

  // H_m0 by H_2
  colvals[30] = 3 ;

  // H_m0 by H_m0
  colvals[31] = 4 ;

  // H_m0 by de
  colvals[32] = 8 ;

  // H_m0 by ge
  colvals[33] = 9 ;

  // He_1 by He_1
  colvals[34] = 5 ;

  // He_1 by He_2
  colvals[35] = 6 ;

  // He_1 by de
  colvals[36] = 8 ;

  // He_1 by ge
  colvals[37] = 9 ;

  // He_2 by He_1
  colvals[38] = 5 ;

  // He_2 by He_2
  colvals[39] = 6 ;

  // He_2 by He_3
  colvals[40] = 7 ;

  // He_2 by de
  colvals[41] = 8 ;

  // He_2 by ge
  colvals[42] = 9 ;

  // He_3 by He_2
  colvals[43] = 6 ;

  // He_3 by He_3
  colvals[44] = 7 ;

  // He_3 by de
  colvals[45] = 8 ;

  // He_3 by ge
  colvals[46] = 9 ;

  // de by H2_2
  colvals[47] = 1 ;

  // de by H_1
  colvals[48] = 2 ;

  // de by H_2
  colvals[49] = 3 ;

  // de by H_m0
  colvals[50] = 4 ;

  // de by He_1
  colvals[51] = 5 ;

  // de by He_2
  colvals[52] = 6 ;

  // de by He_3
  colvals[53] = 7 ;

  // de by de
  colvals[54] = 8 ;

  // de by ge
  colvals[55] = 9 ;

  // ge by H2_1
  colvals[56] = 0 ;

  // ge by H_1
  colvals[57] = 2 ;

  // ge by H_2
  colvals[58] = 3 ;

  // ge by He_1
  colvals[59] = 5 ;

  // ge by He_2
  colvals[60] = 6 ;

  // ge by He_3
  colvals[61] = 7 ;

  // ge by de
  colvals[62] = 8 ;

  // ge by ge
  colvals[63] = 9 ;

  // set row pointers for CSR structure
  rowptrs[0] = 0;
  rowptrs[1] = 7;
  rowptrs[2] = 14;
  rowptrs[3] = 21;
  rowptrs[4] = 28;
  rowptrs[5] = 34;
  rowptrs[6] = 38;
  rowptrs[7] = 43;
  rowptrs[8] = 47;
  rowptrs[9] = 56;
  rowptrs[10] = NSPARSE;

  // copy rowptrs, colvals to the device
  SUNMatrix_cuSparse_CopyToDevice(J, NULL, rowptrs, colvals);
  cudaDeviceSynchronize();
#endif

  return 0;
}
#endif



int calculate_jacobian_cvklu(realtype t, N_Vector y, N_Vector fy,
                             SUNMatrix J, void *user_data,
                             N_Vector tmp1, N_Vector tmp2,
                             N_Vector tmp3)
{
  cvklu_data *data    = (cvklu_data*) user_data;
  const double z      = data->current_z;
  double *scale       = data->scale;
  double *inv_scale   = data->inv_scale;
  const double *ydata = N_VGetDeviceArrayPointer(y);
  long int nstrip     = data->nstrip;

  // Access dense matrix structures, and zero out data
#ifdef USEMAGMA
  realtype *matrix_data = SUNMatrix_MagmaDense_Data(J);
#else
  // Access CSR sparse matrix structures, and zero out data
#ifdef RAJA_CUDA
  realtype *matrix_data = SUNMatrix_cuSparse_Data(J);
  sunindextype *rowptrs = SUNMatrix_cuSparse_IndexPointers(J);
  sunindextype *colvals = SUNMatrix_cuSparse_IndexValues(J);
#elif RAJA_SERIAL
  realtype *matrix_data = SUNSparseMatrix_Data(J);
  sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(J);
  sunindextype *colvals = SUNSparseMatrix_IndexValues(J);
#endif
#endif
  SUNMatZero(J);

  // Loop over data, filling in Jacobian
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,nstrip), [=] RAJA_DEVICE (long int i) {

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

    long int idx;

    // H2_1 by H2_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,0), DENSEIDX(i,0,0));
    matrix_data[ idx ] = -k11*H_2 - k12*de - k13*H_1 + k21*pow(H_1, 2);
    matrix_data[ idx ] *=  (inv_scale[ j + 0 ]*scale[ j + 0 ]);

    // H2_1 by H2_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,1), DENSEIDX(i,0,1));
    matrix_data[ idx ] = k10*H_1 + k19*H_m0;
    matrix_data[ idx ] *=  (inv_scale[ j + 0 ]*scale[ j + 1 ]);

    // H2_1 by H_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,2), DENSEIDX(i,0,2));
    matrix_data[ idx ] = k08*H_m0 + k10*H2_2 - k13*H2_1 + 2*k21*H2_1*H_1 + 3*k22*pow(H_1, 2);
    matrix_data[ idx ] *=  (inv_scale[ j + 0 ]*scale[ j + 2 ]);

    // H2_1 by H_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,3), DENSEIDX(i,0,3));
    matrix_data[ idx ] = -k11*H2_1;
    matrix_data[ idx ] *= (inv_scale[ j + 0 ]*scale[ j + 3 ]);

    // H2_1 by H_m0
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,4), DENSEIDX(i,0,4));
    matrix_data[ idx ] = k08*H_1 + k19*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 0 ]*scale[ j + 4 ]);

    // H2_1 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,5), DENSEIDX(i,0,8));
    matrix_data[ idx ] = -k12*H2_1;
    matrix_data[ idx ] *= (inv_scale[ j + 0 ]*scale[ j + 8 ]);

    // H2_1 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,6), DENSEIDX(i,0,9));
    matrix_data[ idx ] = rk08*H_1*H_m0 + rk10*H2_2*H_1 - rk11*H2_1*H_2 - rk12*H2_1*de - rk13*H2_1*H_1 + rk19*H2_2*H_m0 + rk21*H2_1*H_1*H_1 + rk22*H_1*H_1*H_1;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 0 ]*scale[ j + 9 ]);


    // H2_2 by H2_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,7), DENSEIDX(i,1,0));
    matrix_data[ idx ] = k11*H_2;
    matrix_data[ idx ] *= (inv_scale[ j + 1 ]*scale[ j + 0 ]);

    // H2_2 by H2_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,8), DENSEIDX(i,1,1));
    matrix_data[ idx ] = -k10*H_1 - k18*de - k19*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 1 ]*scale[ j + 1 ]);

    // H2_2 by H_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,9), DENSEIDX(i,1,2));
    matrix_data[ idx ] = k09*H_2 - k10*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 1 ]*scale[ j + 2 ]);

    // H2_2 by H_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,10), DENSEIDX(i,1,3));
    matrix_data[ idx ] = k09*H_1 + k11*H2_1 + k17*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 1 ]*scale[ j + 3 ]);

    // H2_2 by H_m0
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,11), DENSEIDX(i,1,4));
    matrix_data[ idx ] = k17*H_2 - k19*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 1 ]*scale[ j + 4 ]);

    // H2_2 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,12), DENSEIDX(i,1,8));
    matrix_data[ idx ] = -k18*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 1 ]*scale[ j + 8 ]);

    // H2_2 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,13), DENSEIDX(i,1,9));
    matrix_data[ idx ] = rk09*H_1*H_2 - rk10*H2_2*H_1 + rk11*H2_1*H_2 + rk17*H_2*H_m0 - rk18*H2_2*de - rk19*H2_2*H_m0;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 1 ]*scale[ j + 9 ]);


    // H_1 by H2_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,14), DENSEIDX(i,2,0));
    matrix_data[ idx ] = k11*H_2 + 2*k12*de + 2*k13*H_1 - 2*k21*pow(H_1, 2);
    matrix_data[ idx ] *= (inv_scale[ j + 2 ]*scale[ j + 0 ]);

    // H_1 by H2_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,15), DENSEIDX(i,2,1));
    matrix_data[ idx ] = -k10*H_1 + 2*k18*de + k19*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 2 ]*scale[ j + 1 ]);

    // H_1 by H_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,16), DENSEIDX(i,2,2));
    matrix_data[ idx ] = -k01*de - k07*de - k08*H_m0 - k09*H_2 - k10*H2_2 + 2*k13*H2_1 + k15*H_m0 - 4*k21*H2_1*H_1 - 6*k22*pow(H_1, 2);
    matrix_data[ idx ]  *= (inv_scale[ j + 2 ]*scale[ j + 2 ]);

    // H_1 by H_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,17), DENSEIDX(i,2,3));
    matrix_data[ idx ] = k02*de - k09*H_1 + k11*H2_1 + 2*k16*H_m0;
    matrix_data[ idx ]  *= (inv_scale[ j + 2 ]*scale[ j + 3 ]);

    // H_1 by H_m0
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,18), DENSEIDX(i,2,4));
    matrix_data[ idx ] = -k08*H_1 + k14*de + k15*H_1 + 2*k16*H_2 + k19*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 2 ]*scale[ j + 4 ]);

    // H_1 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,19), DENSEIDX(i,2,8));
    matrix_data[ idx ] = -k01*H_1 + k02*H_2 - k07*H_1 + 2*k12*H2_1 + k14*H_m0 + 2*k18*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 2 ]*scale[ j + 8 ]);

    // H_1 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,20), DENSEIDX(i,2,9));
    matrix_data[ idx ] = -rk01*H_1*de + rk02*H_2*de - rk07*H_1*de - rk08*H_1*H_m0 - rk09*H_1*H_2 - rk10*H2_2*H_1 + rk11*H2_1*H_2 + 2*rk12*H2_1*de + 2*rk13*H2_1*H_1 + rk14*H_m0*de + rk15*H_1*H_m0 + 2*rk16*H_2*H_m0 + 2*rk18*H2_2*de + rk19*H2_2*H_m0 - 2*rk21*H2_1*H_1*H_1 - 2*rk22*H_1*H_1*H_1;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 2 ]*scale[ j + 9 ]);


    // H_2 by H2_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,21), DENSEIDX(i,3,0));
    matrix_data[ idx ] = -k11*H_2;
    matrix_data[ idx ] *= (inv_scale[ j + 3 ]*scale[ j + 0 ]);

    // H_2 by H2_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,22), DENSEIDX(i,3,1));
    matrix_data[ idx ] = k10*H_1;
    matrix_data[ idx ] *= (inv_scale[ j + 3 ]*scale[ j + 1 ]);

    // H_2 by H_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,23), DENSEIDX(i,3,2));
    matrix_data[ idx ] = k01*de - k09*H_2 + k10*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 3 ]*scale[ j + 2 ]);

    // H_2 by H_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,24), DENSEIDX(i,3,3));
    matrix_data[ idx ] = -k02*de - k09*H_1 - k11*H2_1 - k16*H_m0 - k17*H_m0;
    matrix_data[ idx ]  *= (inv_scale[ j + 3 ]*scale[ j + 3 ]);

    // H_2 by H_m0
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,25), DENSEIDX(i,3,4));
    matrix_data[ idx ] = -k16*H_2 - k17*H_2;
    matrix_data[ idx ] *= (inv_scale[ j + 3 ]*scale[ j + 4 ]);

    // H_2 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,26), DENSEIDX(i,3,8));
    matrix_data[ idx ] = k01*H_1 - k02*H_2;
    matrix_data[ idx ] *= (inv_scale[ j + 3 ]*scale[ j + 8 ]);

    // H_2 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,27), DENSEIDX(i,3,9));
    matrix_data[ idx ] = rk01*H_1*de - rk02*H_2*de - rk09*H_1*H_2 + rk10*H2_2*H_1 - rk11*H2_1*H_2 - rk16*H_2*H_m0 - rk17*H_2*H_m0;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 3 ]*scale[ j + 9 ]);


    // H_m0 by H2_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,28), DENSEIDX(i,4,1));
    matrix_data[ idx ] = -k19*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 4 ]*scale[ j + 1 ]);

    // H_m0 by H_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,29), DENSEIDX(i,4,2));
    matrix_data[ idx ] = k07*de - k08*H_m0 - k15*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 4 ]*scale[ j + 2 ]);

    // H_m0 by H_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,30), DENSEIDX(i,4,3));
    matrix_data[ idx ] = -k16*H_m0 - k17*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 4 ]*scale[ j + 3 ]);

    // H_m0 by H_m0
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,31), DENSEIDX(i,4,4));
    matrix_data[ idx ] = -k08*H_1 - k14*de - k15*H_1 - k16*H_2 - k17*H_2 - k19*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 4 ]*scale[ j + 4 ]);

    // H_m0 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,32), DENSEIDX(i,4,8));
    matrix_data[ idx ] = k07*H_1 - k14*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 4 ]*scale[ j + 8 ]);

    // H_m0 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,33), DENSEIDX(i,4,9));
    matrix_data[ idx ] = rk07*H_1*de - rk08*H_1*H_m0 - rk14*H_m0*de - rk15*H_1*H_m0 - rk16*H_2*H_m0 - rk17*H_2*H_m0 - rk19*H2_2*H_m0;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 4 ]*scale[ j + 9 ]);


    // He_1 by He_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,34), DENSEIDX(i,5,5));
    matrix_data[ idx ] = -k03*de;
    matrix_data[ idx ] *= (inv_scale[ j + 5 ]*scale[ j + 5 ]);

    // He_1 by He_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,35), DENSEIDX(i,5,6));
    matrix_data[ idx ] = k04*de;
    matrix_data[ idx ] *= (inv_scale[ j + 5 ]*scale[ j + 6 ]);

    // He_1 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,36), DENSEIDX(i,5,8));
    matrix_data[ idx ] = -k03*He_1 + k04*He_2;
    matrix_data[ idx ] *= (inv_scale[ j + 5 ]*scale[ j + 8 ]);

    // He_1 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,37), DENSEIDX(i,5,9));
    matrix_data[ idx ] = -rk03*He_1*de + rk04*He_2*de;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 5 ]*scale[ j + 9 ]);


    // He_2 by He_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,38), DENSEIDX(i,6,5));
    matrix_data[ idx ] = k03*de;
    matrix_data[ idx ] *= (inv_scale[ j + 6 ]*scale[ j + 5 ]);

    // He_2 by He_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,39), DENSEIDX(i,6,6));
    matrix_data[ idx ] = -k04*de - k05*de;
    matrix_data[ idx ] *= (inv_scale[ j + 6 ]*scale[ j + 6 ]);

    // He_2 by He_3
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,40), DENSEIDX(i,6,7));
    matrix_data[ idx ] = k06*de;
    matrix_data[ idx ] *= (inv_scale[ j + 6 ]*scale[ j + 7 ]);

    // He_2 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,41), DENSEIDX(i,6,8));
    matrix_data[ idx ] = k03*He_1 - k04*He_2 - k05*He_2 + k06*He_3;
    matrix_data[ idx ] *= (inv_scale[ j + 6 ]*scale[ j + 8 ]);

    // He_2 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,42), DENSEIDX(i,6,9));
    matrix_data[ idx ] = rk03*He_1*de - rk04*He_2*de - rk05*He_2*de + rk06*He_3*de;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 6 ]*scale[ j + 9 ]);


    // He_3 by He_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,43), DENSEIDX(i,7,6));
    matrix_data[ idx ] = k05*de;
    matrix_data[ idx ] *= (inv_scale[ j + 7 ]*scale[ j + 6 ]);

    // He_3 by He_3
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,44), DENSEIDX(i,7,7));
    matrix_data[ idx ] = -k06*de;
    matrix_data[ idx ] *= (inv_scale[ j + 7 ]*scale[ j + 7 ]);

    // He_3 by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,45), DENSEIDX(i,7,8));
    matrix_data[ idx ] = k05*He_2 - k06*He_3;
    matrix_data[ idx ] *= (inv_scale[ j + 7 ]*scale[ j + 8 ]);

    // He_3 by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,46), DENSEIDX(i,7,9));
    matrix_data[ idx ] = rk05*He_2*de - rk06*He_3*de;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 7 ]*scale[ j + 9 ]);


    // de by H2_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,47), DENSEIDX(i,8,1));
    matrix_data[ idx ] = -k18*de;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 1 ]);

    // de by H_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,48), DENSEIDX(i,8,2));
    matrix_data[ idx ] = k01*de - k07*de + k08*H_m0 + k15*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 2 ]);

    // de by H_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,49), DENSEIDX(i,8,3));
    matrix_data[ idx ] = -k02*de + k17*H_m0;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 3 ]);

    // de by H_m0
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,50), DENSEIDX(i,8,4));
    matrix_data[ idx ] = k08*H_1 + k14*de + k15*H_1 + k17*H_2;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 4 ]);

    // de by He_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,51), DENSEIDX(i,8,5));
    matrix_data[ idx ] = k03*de;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 5 ]);

    // de by He_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,52), DENSEIDX(i,8,6));
    matrix_data[ idx ] = -k04*de + k05*de;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 6 ]);

    // de by He_3
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,53), DENSEIDX(i,8,7));
    matrix_data[ idx ] = -k06*de;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 7 ]);

    // de by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,54), DENSEIDX(i,8,8));
    matrix_data[ idx ] = k01*H_1 - k02*H_2 + k03*He_1 - k04*He_2 + k05*He_2 - k06*He_3 - k07*H_1 + k14*H_m0 - k18*H2_2;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 8 ]);

    // de by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,55), DENSEIDX(i,8,9));
    matrix_data[ idx ] = rk01*H_1*de - rk02*H_2*de + rk03*He_1*de - rk04*He_2*de + rk05*He_2*de - rk06*He_3*de - rk07*H_1*de + rk08*H_1*H_m0 + rk14*H_m0*de + rk15*H_1*H_m0 + rk17*H_2*H_m0 - rk18*H2_2*de;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 8 ]*scale[ j + 9 ]);


    // ge by H2_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,56), DENSEIDX(i,9,0));
    matrix_data[ idx ] = -H2_1*gloverabel08_gaH2*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - 0.5*H_1*h2formation_h2mcool*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0) - 2.0158800000000001*cie_cooling_cieco*mdensity - gloverabel08_h2lte*h2_optical_depth_approx/(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0) + 0.5*h2formation_ncrd2*h2formation_ncrn*pow(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat)/pow(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1, 2);
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 0 ]);

    // ge by H_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,57), DENSEIDX(i,9,2));
    matrix_data[ idx ] = -H2_1*gloverabel08_gaHI*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - ceHI_ceHI*de - ciHI_ciHI*de + 0.5*h2formation_ncrd1*h2formation_ncrn*pow(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat)/pow(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1, 2) + 0.5*(-H2_1*h2formation_h2mcool + 3*pow(H_1, 2)*h2formation_h2mheat)*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0);
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 2 ]);

    // ge by H_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,58), DENSEIDX(i,9,3));
    matrix_data[ idx ] = -H2_1*gloverabel08_gaHp*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - brem_brem*de - de*reHII_reHII;
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 3 ]);

    // ge by He_1
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,59), DENSEIDX(i,9,5));
    matrix_data[ idx ] = -H2_1*gloverabel08_gaHe*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - ciHeI_ciHeI*de;
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 5 ]);

    // ge by He_2
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,60), DENSEIDX(i,9,6));
    matrix_data[ idx ] = -brem_brem*de - ceHeII_ceHeII*de - ceHeI_ceHeI*pow(de, 2) - ciHeII_ciHeII*de - ciHeIS_ciHeIS*pow(de, 2) - de*reHeII1_reHeII1 - de*reHeII2_reHeII2;
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 6 ]);

    // ge by He_3
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,61), DENSEIDX(i,9,7));
    matrix_data[ idx ] = -4.0*brem_brem*de - de*reHeIII_reHeIII;
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 7 ]);

    // ge by de
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,62), DENSEIDX(i,9,8));
    matrix_data[ idx ] = -H2_1*gloverabel08_gael*pow(gloverabel08_h2lte, 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2)) - H_1*ceHI_ceHI - H_1*ciHI_ciHI - H_2*reHII_reHII - He_1*ciHeI_ciHeI - He_2*ceHeII_ceHeII - 2*He_2*ceHeI_ceHeI*de - He_2*ciHeII_ciHeII - 2*He_2*ciHeIS_ciHeIS*de - He_2*reHeII1_reHeII1 - He_2*reHeII2_reHeII2 - He_3*reHeIII_reHeIII - brem_brem*(H_2 + He_2 + 4.0*He_3) - compton_comp_*pow(z + 1.0, 4)*(T - 2.73*z - 2.73);
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 8 ]);

    // ge by ge
    idx = SPARSE_OR_DENSE(SPARSEIDX(i,63), DENSEIDX(i,9,9));
    matrix_data[ idx ] = -2.0158800000000001*H2_1*cie_cooling_cieco*cie_optical_depth_approx*mdensity - H2_1*cie_optical_depth_approx*gloverabel08_h2lte*h2_optical_depth_approx/(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0) - H_1*ceHI_ceHI*cie_optical_depth_approx*de - H_1*ciHI_ciHI*cie_optical_depth_approx*de - H_2*cie_optical_depth_approx*de*reHII_reHII - He_1*ciHeI_ciHeI*cie_optical_depth_approx*de - He_2*ceHeII_ceHeII*cie_optical_depth_approx*de - He_2*ceHeI_ceHeI*cie_optical_depth_approx*pow(de, 2) - He_2*ciHeII_ciHeII*cie_optical_depth_approx*de - He_2*ciHeIS_ciHeIS*cie_optical_depth_approx*pow(de, 2) - He_2*cie_optical_depth_approx*de*reHeII1_reHeII1 - He_2*cie_optical_depth_approx*de*reHeII2_reHeII2 - He_3*cie_optical_depth_approx*de*reHeIII_reHeIII - brem_brem*cie_optical_depth_approx*de*(H_2 + He_2 + 4.0*He_3) - cie_optical_depth_approx*compton_comp_*de*pow(z + 1.0, 4)*(T - 2.73*z - 2.73) + 0.5*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat);

    // ad-hoc extra term of f_ge by ge
    // considering ONLY the h2formation/ and continuum cooling
    matrix_data[ idx ] = -H2_1*gloverabel08_h2lte*h2_optical_depth_approx*(-gloverabel08_h2lte*(-H2_1*rgloverabel08_gaH2 - H_1*rgloverabel08_gaHI - H_2*rgloverabel08_gaHp - He_1*rgloverabel08_gaHe - de*rgloverabel08_gael)/pow(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael, 2) - rgloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael))/pow(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0, 2) - H2_1*h2_optical_depth_approx*rgloverabel08_h2lte/(gloverabel08_h2lte/(H2_1*gloverabel08_gaH2 + H_1*gloverabel08_gaHI + H_2*gloverabel08_gaHp + He_1*gloverabel08_gaHe + de*gloverabel08_gael) + 1.0) + 0.5*pow(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool + pow(H_1, 3)*h2formation_h2mheat)*(-1.0*h2formation_ncrn*(-H2_1*rh2formation_ncrd2 - H_1*rh2formation_ncrd1)/pow(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1, 2) - 1.0*rh2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1)) + 0.5*1.0/(h2formation_ncrn/(H2_1*h2formation_ncrd2 + H_1*h2formation_ncrd1) + 1.0)*(-H2_1*H_1*rh2formation_h2mcool + pow(H_1, 3)*rh2formation_h2mheat);
    matrix_data[ idx ] *= inv_mdensity;
    matrix_data[ idx ] *= Tge;
    matrix_data[ idx ] *= (inv_scale[ j + 9 ]*scale[ j + 9 ]);

#if defined(RAJA_SERIAL) && !defined(USEMAGMA)
    colvals[i * NSPARSE + 0] = i * NSPECIES + 0 ;
    colvals[i * NSPARSE + 1] = i * NSPECIES + 1 ;
    colvals[i * NSPARSE + 2] = i * NSPECIES + 2 ;
    colvals[i * NSPARSE + 3] = i * NSPECIES + 3 ;
    colvals[i * NSPARSE + 4] = i * NSPECIES + 4 ;
    colvals[i * NSPARSE + 5] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 6] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 7] = i * NSPECIES + 0 ;
    colvals[i * NSPARSE + 8] = i * NSPECIES + 1 ;
    colvals[i * NSPARSE + 9] = i * NSPECIES + 2 ;
    colvals[i * NSPARSE + 10] = i * NSPECIES + 3 ;
    colvals[i * NSPARSE + 11] = i * NSPECIES + 4 ;
    colvals[i * NSPARSE + 12] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 13] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 14] = i * NSPECIES + 0 ;
    colvals[i * NSPARSE + 15] = i * NSPECIES + 1 ;
    colvals[i * NSPARSE + 16] = i * NSPECIES + 2 ;
    colvals[i * NSPARSE + 17] = i * NSPECIES + 3 ;
    colvals[i * NSPARSE + 18] = i * NSPECIES + 4 ;
    colvals[i * NSPARSE + 19] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 20] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 21] = i * NSPECIES + 0 ;
    colvals[i * NSPARSE + 22] = i * NSPECIES + 1 ;
    colvals[i * NSPARSE + 23] = i * NSPECIES + 2 ;
    colvals[i * NSPARSE + 24] = i * NSPECIES + 3 ;
    colvals[i * NSPARSE + 25] = i * NSPECIES + 4 ;
    colvals[i * NSPARSE + 26] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 27] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 28] = i * NSPECIES + 1 ;
    colvals[i * NSPARSE + 29] = i * NSPECIES + 2 ;
    colvals[i * NSPARSE + 30] = i * NSPECIES + 3 ;
    colvals[i * NSPARSE + 31] = i * NSPECIES + 4 ;
    colvals[i * NSPARSE + 32] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 33] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 34] = i * NSPECIES + 5 ;
    colvals[i * NSPARSE + 35] = i * NSPECIES + 6 ;
    colvals[i * NSPARSE + 36] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 37] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 38] = i * NSPECIES + 5 ;
    colvals[i * NSPARSE + 39] = i * NSPECIES + 6 ;
    colvals[i * NSPARSE + 40] = i * NSPECIES + 7 ;
    colvals[i * NSPARSE + 41] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 42] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 43] = i * NSPECIES + 6 ;
    colvals[i * NSPARSE + 44] = i * NSPECIES + 7 ;
    colvals[i * NSPARSE + 45] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 46] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 47] = i * NSPECIES + 1 ;
    colvals[i * NSPARSE + 48] = i * NSPECIES + 2 ;
    colvals[i * NSPARSE + 49] = i * NSPECIES + 3 ;
    colvals[i * NSPARSE + 50] = i * NSPECIES + 4 ;
    colvals[i * NSPARSE + 51] = i * NSPECIES + 5 ;
    colvals[i * NSPARSE + 52] = i * NSPECIES + 6 ;
    colvals[i * NSPARSE + 53] = i * NSPECIES + 7 ;
    colvals[i * NSPARSE + 54] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 55] = i * NSPECIES + 9 ;
    colvals[i * NSPARSE + 56] = i * NSPECIES + 0 ;
    colvals[i * NSPARSE + 57] = i * NSPECIES + 2 ;
    colvals[i * NSPARSE + 58] = i * NSPECIES + 3 ;
    colvals[i * NSPARSE + 59] = i * NSPECIES + 5 ;
    colvals[i * NSPARSE + 60] = i * NSPECIES + 6 ;
    colvals[i * NSPARSE + 61] = i * NSPECIES + 7 ;
    colvals[i * NSPARSE + 62] = i * NSPECIES + 8 ;
    colvals[i * NSPARSE + 63] = i * NSPECIES + 9 ;

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
    if (i == nstrip-1) {
      rowptrs[ nstrip * NSPECIES ] = nstrip * NSPARSE;
    }
#endif

  });

  // synchronize device memory
  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )
  HIP_OR_CUDA( hipError_t cuerr = hipGetLastError();,
               cudaError_t cuerr = cudaGetLastError(); )
  if (cuerr != HIP_OR_CUDA( hipSuccess, cudaSuccess )) {
    std::cerr << ">>> ERROR in calculate_jacobian_cvklu: XGetLastError returned %s\n"
              << HIP_OR_CUDA( hipGetErrorName(cuerr), cudaGetErrorName(cuerr) );
    return(-1);
  }

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
