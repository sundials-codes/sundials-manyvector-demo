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


cvklu_data *cvklu_setup_data(const char *FileLocation, int ncells)
{

  //-----------------------------------------------------
  // Function : cvklu_setup_data
  // Description: Initialize a data object that stores the reaction/ cooling rate data
  //-----------------------------------------------------

  int i, n;

  cvklu_data *data = (cvklu_data *) malloc(sizeof(cvklu_data));

  // point the module to look for cvklu_tables.h5
  data->dengo_data_file = FileLocation;

  /* allocate space for the scale related pieces */
  data->Ts = (double *) malloc(ncells*sizeof(double));
  data->dTs_ge = (double *) malloc(ncells*sizeof(double));
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
  data->cs_gammah_gammah = (double *) malloc(ncells*sizeof(double));
  data->dcs_gammah_gammah = (double *) malloc(ncells*sizeof(double));
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
  data->cs_gloverabel08_gphdl = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_gphdl = (double *) malloc(ncells*sizeof(double));
  data->cs_gloverabel08_gpldl = (double *) malloc(ncells*sizeof(double));
  data->dcs_gloverabel08_gpldl = (double *) malloc(ncells*sizeof(double));
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
  data->mdensity = (double *) malloc(ncells*sizeof(double));
  data->inv_mdensity = (double *) malloc(ncells*sizeof(double));
  data->cie_optical_depth_approx = (double *) malloc(ncells*sizeof(double));
  data->h2_optical_depth_approx = (double *) malloc(ncells*sizeof(double));
  data->scale = (double *) malloc(NSPECIES*ncells*sizeof(double));
  data->inv_scale = (double *) malloc(NSPECIES*ncells*sizeof(double));
  data->current_z = 0.0;

  // Number of cells to be solved in a batch
  data->nstrip = ncells;
  /*initialize temperature so it wont crash*/
  for ( i = 0; i < ncells; i++ ) {
    data->Ts[i]    = 1000.0;
  }

  /* Temperature-related pieces */
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


void cvklu_read_rate_tables(cvklu_data *data)
{
  const char * filedir;
  if (data->dengo_data_file != NULL){
    filedir =  data->dengo_data_file;
  } else{
    filedir = "cvklu_tables.h5";
  }

  hid_t file_id = H5Fopen( filedir , H5F_ACC_RDONLY, H5P_DEFAULT);
  /* Allocate the correct number of rate tables */
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
  /* Allocate the correct number of rate tables */
  H5LTread_dataset_double(file_id, "/brem_brem",
                          data->c_brem_brem);
  H5LTread_dataset_double(file_id, "/ceHeI_ceHeI",
                          data->c_ceHeI_ceHeI);
  H5LTread_dataset_double(file_id, "/ceHeII_ceHeII",
                          data->c_ceHeII_ceHeII);
  H5LTread_dataset_double(file_id, "/ceHI_ceHI",
                          data->c_ceHI_ceHI);
  H5LTread_dataset_double(file_id, "/cie_cooling_cieco",
                          data->c_cie_cooling_cieco);
  H5LTread_dataset_double(file_id, "/ciHeI_ciHeI",
                          data->c_ciHeI_ciHeI);
  H5LTread_dataset_double(file_id, "/ciHeII_ciHeII",
                          data->c_ciHeII_ciHeII);
  H5LTread_dataset_double(file_id, "/ciHeIS_ciHeIS",
                          data->c_ciHeIS_ciHeIS);
  H5LTread_dataset_double(file_id, "/ciHI_ciHI",
                          data->c_ciHI_ciHI);
  H5LTread_dataset_double(file_id, "/compton_comp_",
                          data->c_compton_comp_);
  H5LTread_dataset_double(file_id, "/gammah_gammah",
                          data->c_gammah_gammah);
  H5LTread_dataset_double(file_id, "/gloverabel08_gael",
                          data->c_gloverabel08_gael);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaH2",
                          data->c_gloverabel08_gaH2);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHe",
                          data->c_gloverabel08_gaHe);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHI",
                          data->c_gloverabel08_gaHI);
  H5LTread_dataset_double(file_id, "/gloverabel08_gaHp",
                          data->c_gloverabel08_gaHp);
  H5LTread_dataset_double(file_id, "/gloverabel08_gphdl",
                          data->c_gloverabel08_gphdl);
  H5LTread_dataset_double(file_id, "/gloverabel08_gpldl",
                          data->c_gloverabel08_gpldl);
  H5LTread_dataset_double(file_id, "/gloverabel08_h2lte",
                          data->c_gloverabel08_h2lte);
  H5LTread_dataset_double(file_id, "/h2formation_h2mcool",
                          data->c_h2formation_h2mcool);
  H5LTread_dataset_double(file_id, "/h2formation_h2mheat",
                          data->c_h2formation_h2mheat);
  H5LTread_dataset_double(file_id, "/h2formation_ncrd1",
                          data->c_h2formation_ncrd1);
  H5LTread_dataset_double(file_id, "/h2formation_ncrd2",
                          data->c_h2formation_ncrd2);
  H5LTread_dataset_double(file_id, "/h2formation_ncrn",
                          data->c_h2formation_ncrn);
  H5LTread_dataset_double(file_id, "/reHeII1_reHeII1",
                          data->c_reHeII1_reHeII1);
  H5LTread_dataset_double(file_id, "/reHeII2_reHeII2",
                          data->c_reHeII2_reHeII2);
  H5LTread_dataset_double(file_id, "/reHeIII_reHeIII",
                          data->c_reHeIII_reHeIII);
  H5LTread_dataset_double(file_id, "/reHII_reHII",
                          data->c_reHII_reHII);

  H5Fclose(file_id);
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
  /* Allocate the correct number of rate tables */
  H5LTread_dataset_double(file_id, "/gammaH2_1",
                          data->g_gammaH2_1 );
  H5LTread_dataset_double(file_id, "/dgammaH2_1_dT",
                          data->g_dgammaH2_1_dT );

  H5LTread_dataset_double(file_id, "/gammaH2_2",
                          data->g_gammaH2_2 );
  H5LTread_dataset_double(file_id, "/dgammaH2_2_dT",
                          data->g_dgammaH2_2_dT );


  H5Fclose(file_id);

}



int cvklu_calculate_temperature(cvklu_data *data,
                                double *input, int nstrip, int nchem)
{
  int i, j;
  double density, T, Tnew;
  double kb = 1.3806504e-16; // Boltzmann constant [erg/K]
  double mh = 1.67e-24;
  double gamma = 5.e0/3.e0;
  double _gamma_m1 = 1.0 / (gamma - 1);


  double gammaH2 = 7.e0/5.e0; // Should be a function of temperature
  // this is a temporary solution
  double dge_dT;
  double dge;


  double gammaH2_1;
  double dgammaH2_1_dT;
  double _gammaH2_1_m1;

  double gammaH2_2;
  double dgammaH2_2_dT;
  double _gammaH2_2_m1;


  double Tdiff = 1.0;
  int MAX_T_ITERATION = 100;
  int count = 0;





  /* Calculate total density */
  double H2_1;
  double H2_2;
  double H_1;
  double H_2;
  double H_m0;
  double He_1;
  double He_2;
  double He_3;
  double de;
  double ge;


  i = 0;

  for ( i = 0; i < nstrip; i++ ){
    j = i * nchem;
    H2_1 = input[j];
    j++;

    H2_2 = input[j];
    j++;

    H_1 = input[j];
    j++;

    H_2 = input[j];
    j++;

    H_m0 = input[j];
    j++;

    He_1 = input[j];
    j++;

    He_2 = input[j];
    j++;

    He_3 = input[j];
    j++;

    de = input[j];
    j++;

    ge = input[j];
    j++;



    density = 2.0*H2_1 + 2.0*H2_2 + 1.0079400000000001*H_1 + 1.0079400000000001*H_2 + 1.0079400000000001*H_m0 + 4.0026020000000004*He_1 + 4.0026020000000004*He_2 + 4.0026020000000004*He_3;




    // Initiate the "guess" temperature
    T    = data->Ts[i];
    Tnew = T*1.1;

    Tdiff = Tnew - T;
    count = 0;

    //        while ( Tdiff/ Tnew > 0.001 ){
    for (j=0; j<10; j++){
      // We do Newton's Iteration to calculate the temperature
      // Since gammaH2 is dependent on the temperature too!

      T = data->Ts[i];

      cvklu_interpolate_gamma(data, i);

      gammaH2_1 = data->gammaH2_1[i];
      dgammaH2_1_dT = data->dgammaH2_1_dT[i];
      _gammaH2_1_m1 = 1.0 / (gammaH2_1 - 1.0);

      gammaH2_2 = data->gammaH2_2[i];
      dgammaH2_2_dT = data->dgammaH2_2_dT[i];
      _gammaH2_2_m1 = 1.0 / (gammaH2_2 - 1.0);

      // update gammaH2
      // The derivatives of  sum (nkT/(gamma - 1)/mh/density) - ge
      // This is the function we want to minimize
      // which should only be dependent on the first part
      dge_dT = T*kb*(-H2_1*_gammaH2_1_m1*_gammaH2_1_m1*dgammaH2_1_dT - H2_2*_gammaH2_2_m1*_gammaH2_2_m1*dgammaH2_2_dT)/(density*mh) + kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh);

      //This is the change in ge for each iteration
      dge = T*kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh) - ge;

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

  } // for i in nstrip loop
  return 0;

}



void cvklu_interpolate_rates(cvklu_data *data, int nstrip)
{
  int i, bin_id, zbin_id;
  double lb, t1, t2;
  double Tdef, dT, invTs, Tfactor;
  lb = log(data->bounds[0]);

  i = 0;

  for ( i = 0; i < nstrip; i++ ){
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

    data->cs_gammah_gammah[i] = data->c_gammah_gammah[bin_id] +
      Tdef * (data->c_gammah_gammah[bin_id+1] - data->c_gammah_gammah[bin_id]);
    data->dcs_gammah_gammah[i] = (data->c_gammah_gammah[bin_id+1] - data->c_gammah_gammah[bin_id])*Tfactor;

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

    data->cs_gloverabel08_gphdl[i] = data->c_gloverabel08_gphdl[bin_id] +
      Tdef * (data->c_gloverabel08_gphdl[bin_id+1] - data->c_gloverabel08_gphdl[bin_id]);
    data->dcs_gloverabel08_gphdl[i] = (data->c_gloverabel08_gphdl[bin_id+1] - data->c_gloverabel08_gphdl[bin_id])*Tfactor;

    data->cs_gloverabel08_gpldl[i] = data->c_gloverabel08_gpldl[bin_id] +
      Tdef * (data->c_gloverabel08_gpldl[bin_id+1] - data->c_gloverabel08_gpldl[bin_id]);
    data->dcs_gloverabel08_gpldl[i] = (data->c_gloverabel08_gpldl[bin_id+1] - data->c_gloverabel08_gpldl[bin_id])*Tfactor;

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


}



void cvklu_interpolate_gamma(cvklu_data *data, int i)
{

  /*
   * find the bin_id for the given temperature
   * update dT for i_th strip
   */

  int bin_id, zbin_id;
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
    Tdef * (data->g_dgammaH2_2_dT[bin_id+1]
                     - data->g_dgammaH2_2_dT[bin_id]);


  data->gammaH2_1[i] = data->g_gammaH2_1[bin_id] +
    Tdef * (data->g_gammaH2_1[bin_id+1] - data->g_gammaH2_1[bin_id]);

  data->dgammaH2_1_dT[i] = data->g_dgammaH2_1_dT[bin_id] +
    Tdef * (data->g_dgammaH2_1_dT[bin_id+1]
                     - data->g_dgammaH2_1_dT[bin_id]);


}






int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  cvklu_data *data = (cvklu_data* ) user_data;
  int i, j;

  int nchem = 10;
  int nstrip = data->nstrip;

  /* change N_Vector back to an array */
  double y_arr[ 10 * nstrip ];
  double H2_1;

  double H2_2;

  double H_1;

  double H_2;

  double H_m0;

  double He_1;

  double He_2;

  double He_3;

  double de;

  double ge;


  double *scale     = data->scale;
  double *inv_scale = data->inv_scale;
  double *ydata     = N_VGetArrayPointer(y);

  for ( i = 0; i < nstrip; i++ ){
    j = i * nchem;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
    y_arr[j] = ydata[j]*scale[j];
    j++;
  }

  int flag;
  flag = cvklu_calculate_temperature(data, y_arr , nstrip, nchem );
  if (flag > 0){
    // check if the temperature failed to converged
    return -1;
  }
  cvklu_interpolate_rates(data, nstrip);


  /* Now we set up some temporaries */
  double *k01 = data->rs_k01;
  double *k02 = data->rs_k02;
  double *k03 = data->rs_k03;
  double *k04 = data->rs_k04;
  double *k05 = data->rs_k05;
  double *k06 = data->rs_k06;
  double *k07 = data->rs_k07;
  double *k08 = data->rs_k08;
  double *k09 = data->rs_k09;
  double *k10 = data->rs_k10;
  double *k11 = data->rs_k11;
  double *k12 = data->rs_k12;
  double *k13 = data->rs_k13;
  double *k14 = data->rs_k14;
  double *k15 = data->rs_k15;
  double *k16 = data->rs_k16;
  double *k17 = data->rs_k17;
  double *k18 = data->rs_k18;
  double *k19 = data->rs_k19;
  double *k21 = data->rs_k21;
  double *k22 = data->rs_k22;
  double *brem_brem = data->cs_brem_brem;
  double *ceHeI_ceHeI = data->cs_ceHeI_ceHeI;
  double *ceHeII_ceHeII = data->cs_ceHeII_ceHeII;
  double *ceHI_ceHI = data->cs_ceHI_ceHI;
  double *cie_cooling_cieco = data->cs_cie_cooling_cieco;
  double *ciHeI_ciHeI = data->cs_ciHeI_ciHeI;
  double *ciHeII_ciHeII = data->cs_ciHeII_ciHeII;
  double *ciHeIS_ciHeIS = data->cs_ciHeIS_ciHeIS;
  double *ciHI_ciHI = data->cs_ciHI_ciHI;
  double *compton_comp_ = data->cs_compton_comp_;
  double *gammah_gammah = data->cs_gammah_gammah;
  double *gloverabel08_gael = data->cs_gloverabel08_gael;
  double *gloverabel08_gaH2 = data->cs_gloverabel08_gaH2;
  double *gloverabel08_gaHe = data->cs_gloverabel08_gaHe;
  double *gloverabel08_gaHI = data->cs_gloverabel08_gaHI;
  double *gloverabel08_gaHp = data->cs_gloverabel08_gaHp;
  double *gloverabel08_gphdl = data->cs_gloverabel08_gphdl;
  double *gloverabel08_gpldl = data->cs_gloverabel08_gpldl;
  double *gloverabel08_h2lte = data->cs_gloverabel08_h2lte;
  double *h2formation_h2mcool = data->cs_h2formation_h2mcool;
  double *h2formation_h2mheat = data->cs_h2formation_h2mheat;
  double *h2formation_ncrd1 = data->cs_h2formation_ncrd1;
  double *h2formation_ncrd2 = data->cs_h2formation_ncrd2;
  double *h2formation_ncrn = data->cs_h2formation_ncrn;
  double *reHeII1_reHeII1 = data->cs_reHeII1_reHeII1;
  double *reHeII2_reHeII2 = data->cs_reHeII2_reHeII2;
  double *reHeIII_reHeIII = data->cs_reHeIII_reHeIII;
  double *reHII_reHII = data->cs_reHII_reHII;


  double h2_optical_depth_approx;


  double cie_optical_depth_approx;


  double z;
  double T;

  double mh = 1.67e-24;
  double mdensity, inv_mdensity;
  double *ydotdata = N_VGetArrayPointer(ydot);


  for ( i = 0; i < nstrip; i++ ){

    T            = data->Ts[i];
    z            = data->current_z;
    mdensity     = data->mdensity[i];
    inv_mdensity = data->inv_mdensity[i];

    h2_optical_depth_approx = data->h2_optical_depth_approx[i];


    cie_optical_depth_approx = data->cie_optical_depth_approx[i];


    j = i * nchem;
    H2_1 = y_arr[j];
    j++;
    H2_2 = y_arr[j];
    j++;
    H_1 = y_arr[j];
    j++;
    H_2 = y_arr[j];
    j++;
    H_m0 = y_arr[j];
    j++;
    He_1 = y_arr[j];
    j++;
    He_2 = y_arr[j];
    j++;
    He_3 = y_arr[j];
    j++;
    de = y_arr[j];
    j++;
    ge = y_arr[j];
    j++;


    j = i * nchem;
    //
    // Species: H2_1
    //
    ydotdata[j] = k08[i]*H_1*H_m0 + k10[i]*H2_2*H_1 - k11[i]*H2_1*H_2 - k12[i]*H2_1*de - k13[i]*H2_1*H_1 + k19[i]*H2_2*H_m0 + k21[i]*H2_1*H_1*H_1 + k22[i]*H_1*H_1*H_1;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: H2_2
    //
    ydotdata[j] = k09[i]*H_1*H_2 - k10[i]*H2_2*H_1 + k11[i]*H2_1*H_2 + k17[i]*H_2*H_m0 - k18[i]*H2_2*de - k19[i]*H2_2*H_m0;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: H_1
    //
    ydotdata[j] = -k01[i]*H_1*de + k02[i]*H_2*de - k07[i]*H_1*de - k08[i]*H_1*H_m0 - k09[i]*H_1*H_2 - k10[i]*H2_2*H_1 + k11[i]*H2_1*H_2 + 2*k12[i]*H2_1*de + 2*k13[i]*H2_1*H_1 + k14[i]*H_m0*de + k15[i]*H_1*H_m0 + 2*k16[i]*H_2*H_m0 + 2*k18[i]*H2_2*de + k19[i]*H2_2*H_m0 - 2*k21[i]*H2_1*H_1*H_1 - 2*k22[i]*H_1*H_1*H_1;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: H_2
    //
    ydotdata[j] = k01[i]*H_1*de - k02[i]*H_2*de - k09[i]*H_1*H_2 + k10[i]*H2_2*H_1 - k11[i]*H2_1*H_2 - k16[i]*H_2*H_m0 - k17[i]*H_2*H_m0;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: H_m0
    //
    ydotdata[j] = k07[i]*H_1*de - k08[i]*H_1*H_m0 - k14[i]*H_m0*de - k15[i]*H_1*H_m0 - k16[i]*H_2*H_m0 - k17[i]*H_2*H_m0 - k19[i]*H2_2*H_m0;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: He_1
    //
    ydotdata[j] = -k03[i]*He_1*de + k04[i]*He_2*de;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: He_2
    //
    ydotdata[j] = k03[i]*He_1*de - k04[i]*He_2*de - k05[i]*He_2*de + k06[i]*He_3*de;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: He_3
    //
    ydotdata[j] = k05[i]*He_2*de - k06[i]*He_3*de;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: de
    //
    ydotdata[j] = k01[i]*H_1*de - k02[i]*H_2*de + k03[i]*He_1*de - k04[i]*He_2*de + k05[i]*He_2*de - k06[i]*He_3*de - k07[i]*H_1*de + k08[i]*H_1*H_m0 + k14[i]*H_m0*de + k15[i]*H_1*H_m0 + k17[i]*H_2*H_m0 - k18[i]*H2_2*de;
    ydotdata[j] *= inv_scale[j];

    j++;

    //
    // Species: ge
    //
    ydotdata[j] = -2.0158800000000001*H2_1*cie_cooling_cieco[i]*cie_optical_depth_approx*mdensity - H2_1*cie_optical_depth_approx*gloverabel08_h2lte[i]*h2_optical_depth_approx/(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0) - H_1*ceHI_ceHI[i]*cie_optical_depth_approx*de - H_1*ciHI_ciHI[i]*cie_optical_depth_approx*de - H_2*cie_optical_depth_approx*de*reHII_reHII[i] - He_1*ciHeI_ciHeI[i]*cie_optical_depth_approx*de - He_2*ceHeII_ceHeII[i]*cie_optical_depth_approx*de - He_2*ceHeI_ceHeI[i]*cie_optical_depth_approx*pow(de, 2) - He_2*ciHeII_ciHeII[i]*cie_optical_depth_approx*de - He_2*ciHeIS_ciHeIS[i]*cie_optical_depth_approx*pow(de, 2) - He_2*cie_optical_depth_approx*de*reHeII1_reHeII1[i] - He_2*cie_optical_depth_approx*de*reHeII2_reHeII2[i] - He_3*cie_optical_depth_approx*de*reHeIII_reHeIII[i] - brem_brem[i]*cie_optical_depth_approx*de*(H_2 + He_2 + 4.0*He_3) - cie_optical_depth_approx*compton_comp_[i]*de*pow(z + 1.0, 4)*(T - 2.73*z - 2.73) + 0.5*1.0/(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0)*(-H2_1*H_1*h2formation_h2mcool[i] + pow(H_1, 3)*h2formation_h2mheat[i]);
    ydotdata[j] *= inv_scale[j];

    ydotdata[j] *= inv_mdensity;

    j++;

  }
  return 0;
}




int calculate_sparse_jacobian_cvklu( realtype t,
                                     N_Vector y, N_Vector fy,
                                     SUNMatrix J, void *user_data,
                                     N_Vector tmp1, N_Vector tmp2,
                                     N_Vector tmp3)
{
  /* We iterate over all of the rates */
  /* Calcuate temperature first */

  cvklu_data *data = (cvklu_data*)user_data;

  int nchem = 10;
  int nstrip = data->nstrip;
  int i, j;
  int NSPARSE = 64;

  /* change N_Vector back to an array */
  double y_arr[ 10 * nstrip ];
  double *scale     = data->scale;
  double *inv_scale = data->inv_scale;

  double h2_optical_depth_approx;


  double cie_optical_depth_approx;


  /* Now We set up some temporaries */

  // CSR is what we choose
  sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(J);
  sunindextype *colvals = SUNSparseMatrix_IndexValues(J);
  realtype *matrix_data = SUNSparseMatrix_Data(J);

  SUNMatZero(J);

  double *Tge = data->dTs_ge;
  double *k01 = data->rs_k01;
  double *rk01= data->drs_k01;
  double *k02 = data->rs_k02;
  double *rk02= data->drs_k02;
  double *k03 = data->rs_k03;
  double *rk03= data->drs_k03;
  double *k04 = data->rs_k04;
  double *rk04= data->drs_k04;
  double *k05 = data->rs_k05;
  double *rk05= data->drs_k05;
  double *k06 = data->rs_k06;
  double *rk06= data->drs_k06;
  double *k07 = data->rs_k07;
  double *rk07= data->drs_k07;
  double *k08 = data->rs_k08;
  double *rk08= data->drs_k08;
  double *k09 = data->rs_k09;
  double *rk09= data->drs_k09;
  double *k10 = data->rs_k10;
  double *rk10= data->drs_k10;
  double *k11 = data->rs_k11;
  double *rk11= data->drs_k11;
  double *k12 = data->rs_k12;
  double *rk12= data->drs_k12;
  double *k13 = data->rs_k13;
  double *rk13= data->drs_k13;
  double *k14 = data->rs_k14;
  double *rk14= data->drs_k14;
  double *k15 = data->rs_k15;
  double *rk15= data->drs_k15;
  double *k16 = data->rs_k16;
  double *rk16= data->drs_k16;
  double *k17 = data->rs_k17;
  double *rk17= data->drs_k17;
  double *k18 = data->rs_k18;
  double *rk18= data->drs_k18;
  double *k19 = data->rs_k19;
  double *rk19= data->drs_k19;
  double *k21 = data->rs_k21;
  double *rk21= data->drs_k21;
  double *k22 = data->rs_k22;
  double *rk22= data->drs_k22;
  double *brem_brem = data->cs_brem_brem;
  double *rbrem_brem = data->dcs_brem_brem;
  double *ceHeI_ceHeI = data->cs_ceHeI_ceHeI;
  double *rceHeI_ceHeI = data->dcs_ceHeI_ceHeI;
  double *ceHeII_ceHeII = data->cs_ceHeII_ceHeII;
  double *rceHeII_ceHeII = data->dcs_ceHeII_ceHeII;
  double *ceHI_ceHI = data->cs_ceHI_ceHI;
  double *rceHI_ceHI = data->dcs_ceHI_ceHI;
  double *cie_cooling_cieco = data->cs_cie_cooling_cieco;
  double *rcie_cooling_cieco = data->dcs_cie_cooling_cieco;
  double *ciHeI_ciHeI = data->cs_ciHeI_ciHeI;
  double *rciHeI_ciHeI = data->dcs_ciHeI_ciHeI;
  double *ciHeII_ciHeII = data->cs_ciHeII_ciHeII;
  double *rciHeII_ciHeII = data->dcs_ciHeII_ciHeII;
  double *ciHeIS_ciHeIS = data->cs_ciHeIS_ciHeIS;
  double *rciHeIS_ciHeIS = data->dcs_ciHeIS_ciHeIS;
  double *ciHI_ciHI = data->cs_ciHI_ciHI;
  double *rciHI_ciHI = data->dcs_ciHI_ciHI;
  double *compton_comp_ = data->cs_compton_comp_;
  double *rcompton_comp_ = data->dcs_compton_comp_;
  double *gammah_gammah = data->cs_gammah_gammah;
  double *rgammah_gammah = data->dcs_gammah_gammah;
  double *gloverabel08_gael = data->cs_gloverabel08_gael;
  double *rgloverabel08_gael = data->dcs_gloverabel08_gael;
  double *gloverabel08_gaH2 = data->cs_gloverabel08_gaH2;
  double *rgloverabel08_gaH2 = data->dcs_gloverabel08_gaH2;
  double *gloverabel08_gaHe = data->cs_gloverabel08_gaHe;
  double *rgloverabel08_gaHe = data->dcs_gloverabel08_gaHe;
  double *gloverabel08_gaHI = data->cs_gloverabel08_gaHI;
  double *rgloverabel08_gaHI = data->dcs_gloverabel08_gaHI;
  double *gloverabel08_gaHp = data->cs_gloverabel08_gaHp;
  double *rgloverabel08_gaHp = data->dcs_gloverabel08_gaHp;
  double *gloverabel08_gphdl = data->cs_gloverabel08_gphdl;
  double *rgloverabel08_gphdl = data->dcs_gloverabel08_gphdl;
  double *gloverabel08_gpldl = data->cs_gloverabel08_gpldl;
  double *rgloverabel08_gpldl = data->dcs_gloverabel08_gpldl;
  double *gloverabel08_h2lte = data->cs_gloverabel08_h2lte;
  double *rgloverabel08_h2lte = data->dcs_gloverabel08_h2lte;
  double *h2formation_h2mcool = data->cs_h2formation_h2mcool;
  double *rh2formation_h2mcool = data->dcs_h2formation_h2mcool;
  double *h2formation_h2mheat = data->cs_h2formation_h2mheat;
  double *rh2formation_h2mheat = data->dcs_h2formation_h2mheat;
  double *h2formation_ncrd1 = data->cs_h2formation_ncrd1;
  double *rh2formation_ncrd1 = data->dcs_h2formation_ncrd1;
  double *h2formation_ncrd2 = data->cs_h2formation_ncrd2;
  double *rh2formation_ncrd2 = data->dcs_h2formation_ncrd2;
  double *h2formation_ncrn = data->cs_h2formation_ncrn;
  double *rh2formation_ncrn = data->dcs_h2formation_ncrn;
  double *reHeII1_reHeII1 = data->cs_reHeII1_reHeII1;
  double *rreHeII1_reHeII1 = data->dcs_reHeII1_reHeII1;
  double *reHeII2_reHeII2 = data->cs_reHeII2_reHeII2;
  double *rreHeII2_reHeII2 = data->dcs_reHeII2_reHeII2;
  double *reHeIII_reHeIII = data->cs_reHeIII_reHeIII;
  double *rreHeIII_reHeIII = data->dcs_reHeIII_reHeIII;
  double *reHII_reHII = data->cs_reHII_reHII;
  double *rreHII_reHII = data->dcs_reHII_reHII;
  double H2_1;
  double H2_2;
  double H_1;
  double H_2;
  double H_m0;
  double He_1;
  double He_2;
  double He_3;
  double de;
  double ge;
  double z;
  double T;

  double mh = 1.67e-24;
  double mdensity, inv_mdensity;

  double scale2, inv_scale1;
  double *ydata = N_VGetArrayPointer(y);

  j = 0;
  mdensity = 0.0;
  z = data->current_z;

  int k = 0;
  for ( i = 0; i < nstrip; i++ ){
    j = i * nchem;
    H2_1 = ydata[j]*scale[j];
    j++;

    H2_2 = ydata[j]*scale[j];
    j++;

    H_1 = ydata[j]*scale[j];
    j++;

    H_2 = ydata[j]*scale[j];
    j++;

    H_m0 = ydata[j]*scale[j];
    j++;

    He_1 = ydata[j]*scale[j];
    j++;

    He_2 = ydata[j]*scale[j];
    j++;

    He_3 = ydata[j]*scale[j];
    j++;

    de = ydata[j]*scale[j];
    j++;

    ge = ydata[j]*scale[j];
    j++;


    mdensity = data->mdensity[i];
    inv_mdensity = 1.0 / mdensity;

    h2_optical_depth_approx = data->h2_optical_depth_approx[i];



    cie_optical_depth_approx = data->cie_optical_depth_approx[i];


    j = i * NSPARSE;

    // H2_1 by H2_1
    colvals[j + 0] = i * nchem + 0 ;
    matrix_data[ j + 0 ] = -k11[i]*H_2 - k12[i]*de - k13[i]*H_1 + k21[i]*pow(H_1, 2);




    // H2_1 by H2_2
    colvals[j + 1] = i * nchem + 1 ;
    matrix_data[ j + 1 ] = k10[i]*H_1 + k19[i]*H_m0;




    // H2_1 by H_1
    colvals[j + 2] = i * nchem + 2 ;
    matrix_data[ j + 2 ] = k08[i]*H_m0 + k10[i]*H2_2 - k13[i]*H2_1 + 2*k21[i]*H2_1*H_1 + 3*k22[i]*pow(H_1, 2);




    // H2_1 by H_2
    colvals[j + 3] = i * nchem + 3 ;
    matrix_data[ j + 3 ] = -k11[i]*H2_1;




    // H2_1 by H_m0
    colvals[j + 4] = i * nchem + 4 ;
    matrix_data[ j + 4 ] = k08[i]*H_1 + k19[i]*H2_2;




    // H2_1 by de
    colvals[j + 5] = i * nchem + 8 ;
    matrix_data[ j + 5 ] = -k12[i]*H2_1;




    // H2_1 by ge
    colvals[j + 6] = i * nchem + 9 ;
    matrix_data[ j + 6 ] = rk08[i]*H_1*H_m0 + rk10[i]*H2_2*H_1 - rk11[i]*H2_1*H_2 - rk12[i]*H2_1*de - rk13[i]*H2_1*H_1 + rk19[i]*H2_2*H_m0 + rk21[i]*H2_1*H_1*H_1 + rk22[i]*H_1*H_1*H_1;



    matrix_data[ j + 6] *= Tge[i];


    // H2_2 by H2_1
    colvals[j + 7] = i * nchem + 0 ;
    matrix_data[ j + 7 ] = k11[i]*H_2;




    // H2_2 by H2_2
    colvals[j + 8] = i * nchem + 1 ;
    matrix_data[ j + 8 ] = -k10[i]*H_1 - k18[i]*de - k19[i]*H_m0;




    // H2_2 by H_1
    colvals[j + 9] = i * nchem + 2 ;
    matrix_data[ j + 9 ] = k09[i]*H_2 - k10[i]*H2_2;




    // H2_2 by H_2
    colvals[j + 10] = i * nchem + 3 ;
    matrix_data[ j + 10 ] = k09[i]*H_1 + k11[i]*H2_1 + k17[i]*H_m0;




    // H2_2 by H_m0
    colvals[j + 11] = i * nchem + 4 ;
    matrix_data[ j + 11 ] = k17[i]*H_2 - k19[i]*H2_2;




    // H2_2 by de
    colvals[j + 12] = i * nchem + 8 ;
    matrix_data[ j + 12 ] = -k18[i]*H2_2;




    // H2_2 by ge
    colvals[j + 13] = i * nchem + 9 ;
    matrix_data[ j + 13 ] = rk09[i]*H_1*H_2 - rk10[i]*H2_2*H_1 + rk11[i]*H2_1*H_2 + rk17[i]*H_2*H_m0 - rk18[i]*H2_2*de - rk19[i]*H2_2*H_m0;



    matrix_data[ j + 13] *= Tge[i];


    // H_1 by H2_1
    colvals[j + 14] = i * nchem + 0 ;
    matrix_data[ j + 14 ] = k11[i]*H_2 + 2*k12[i]*de + 2*k13[i]*H_1 - 2*k21[i]*pow(H_1, 2);




    // H_1 by H2_2
    colvals[j + 15] = i * nchem + 1 ;
    matrix_data[ j + 15 ] = -k10[i]*H_1 + 2*k18[i]*de + k19[i]*H_m0;




    // H_1 by H_1
    colvals[j + 16] = i * nchem + 2 ;
    matrix_data[ j + 16 ] = -k01[i]*de - k07[i]*de - k08[i]*H_m0 - k09[i]*H_2 - k10[i]*H2_2 + 2*k13[i]*H2_1 + k15[i]*H_m0 - 4*k21[i]*H2_1*H_1 - 6*k22[i]*pow(H_1, 2);




    // H_1 by H_2
    colvals[j + 17] = i * nchem + 3 ;
    matrix_data[ j + 17 ] = k02[i]*de - k09[i]*H_1 + k11[i]*H2_1 + 2*k16[i]*H_m0;




    // H_1 by H_m0
    colvals[j + 18] = i * nchem + 4 ;
    matrix_data[ j + 18 ] = -k08[i]*H_1 + k14[i]*de + k15[i]*H_1 + 2*k16[i]*H_2 + k19[i]*H2_2;




    // H_1 by de
    colvals[j + 19] = i * nchem + 8 ;
    matrix_data[ j + 19 ] = -k01[i]*H_1 + k02[i]*H_2 - k07[i]*H_1 + 2*k12[i]*H2_1 + k14[i]*H_m0 + 2*k18[i]*H2_2;




    // H_1 by ge
    colvals[j + 20] = i * nchem + 9 ;
    matrix_data[ j + 20 ] = -rk01[i]*H_1*de + rk02[i]*H_2*de - rk07[i]*H_1*de - rk08[i]*H_1*H_m0 - rk09[i]*H_1*H_2 - rk10[i]*H2_2*H_1 + rk11[i]*H2_1*H_2 + 2*rk12[i]*H2_1*de + 2*rk13[i]*H2_1*H_1 + rk14[i]*H_m0*de + rk15[i]*H_1*H_m0 + 2*rk16[i]*H_2*H_m0 + 2*rk18[i]*H2_2*de + rk19[i]*H2_2*H_m0 - 2*rk21[i]*H2_1*H_1*H_1 - 2*rk22[i]*H_1*H_1*H_1;



    matrix_data[ j + 20] *= Tge[i];


    // H_2 by H2_1
    colvals[j + 21] = i * nchem + 0 ;
    matrix_data[ j + 21 ] = -k11[i]*H_2;




    // H_2 by H2_2
    colvals[j + 22] = i * nchem + 1 ;
    matrix_data[ j + 22 ] = k10[i]*H_1;




    // H_2 by H_1
    colvals[j + 23] = i * nchem + 2 ;
    matrix_data[ j + 23 ] = k01[i]*de - k09[i]*H_2 + k10[i]*H2_2;




    // H_2 by H_2
    colvals[j + 24] = i * nchem + 3 ;
    matrix_data[ j + 24 ] = -k02[i]*de - k09[i]*H_1 - k11[i]*H2_1 - k16[i]*H_m0 - k17[i]*H_m0;




    // H_2 by H_m0
    colvals[j + 25] = i * nchem + 4 ;
    matrix_data[ j + 25 ] = -k16[i]*H_2 - k17[i]*H_2;




    // H_2 by de
    colvals[j + 26] = i * nchem + 8 ;
    matrix_data[ j + 26 ] = k01[i]*H_1 - k02[i]*H_2;




    // H_2 by ge
    colvals[j + 27] = i * nchem + 9 ;
    matrix_data[ j + 27 ] = rk01[i]*H_1*de - rk02[i]*H_2*de - rk09[i]*H_1*H_2 + rk10[i]*H2_2*H_1 - rk11[i]*H2_1*H_2 - rk16[i]*H_2*H_m0 - rk17[i]*H_2*H_m0;



    matrix_data[ j + 27] *= Tge[i];


    // H_m0 by H2_2
    colvals[j + 28] = i * nchem + 1 ;
    matrix_data[ j + 28 ] = -k19[i]*H_m0;




    // H_m0 by H_1
    colvals[j + 29] = i * nchem + 2 ;
    matrix_data[ j + 29 ] = k07[i]*de - k08[i]*H_m0 - k15[i]*H_m0;




    // H_m0 by H_2
    colvals[j + 30] = i * nchem + 3 ;
    matrix_data[ j + 30 ] = -k16[i]*H_m0 - k17[i]*H_m0;




    // H_m0 by H_m0
    colvals[j + 31] = i * nchem + 4 ;
    matrix_data[ j + 31 ] = -k08[i]*H_1 - k14[i]*de - k15[i]*H_1 - k16[i]*H_2 - k17[i]*H_2 - k19[i]*H2_2;




    // H_m0 by de
    colvals[j + 32] = i * nchem + 8 ;
    matrix_data[ j + 32 ] = k07[i]*H_1 - k14[i]*H_m0;




    // H_m0 by ge
    colvals[j + 33] = i * nchem + 9 ;
    matrix_data[ j + 33 ] = rk07[i]*H_1*de - rk08[i]*H_1*H_m0 - rk14[i]*H_m0*de - rk15[i]*H_1*H_m0 - rk16[i]*H_2*H_m0 - rk17[i]*H_2*H_m0 - rk19[i]*H2_2*H_m0;



    matrix_data[ j + 33] *= Tge[i];


    // He_1 by He_1
    colvals[j + 34] = i * nchem + 5 ;
    matrix_data[ j + 34 ] = -k03[i]*de;




    // He_1 by He_2
    colvals[j + 35] = i * nchem + 6 ;
    matrix_data[ j + 35 ] = k04[i]*de;




    // He_1 by de
    colvals[j + 36] = i * nchem + 8 ;
    matrix_data[ j + 36 ] = -k03[i]*He_1 + k04[i]*He_2;




    // He_1 by ge
    colvals[j + 37] = i * nchem + 9 ;
    matrix_data[ j + 37 ] = -rk03[i]*He_1*de + rk04[i]*He_2*de;



    matrix_data[ j + 37] *= Tge[i];


    // He_2 by He_1
    colvals[j + 38] = i * nchem + 5 ;
    matrix_data[ j + 38 ] = k03[i]*de;




    // He_2 by He_2
    colvals[j + 39] = i * nchem + 6 ;
    matrix_data[ j + 39 ] = -k04[i]*de - k05[i]*de;




    // He_2 by He_3
    colvals[j + 40] = i * nchem + 7 ;
    matrix_data[ j + 40 ] = k06[i]*de;




    // He_2 by de
    colvals[j + 41] = i * nchem + 8 ;
    matrix_data[ j + 41 ] = k03[i]*He_1 - k04[i]*He_2 - k05[i]*He_2 + k06[i]*He_3;




    // He_2 by ge
    colvals[j + 42] = i * nchem + 9 ;
    matrix_data[ j + 42 ] = rk03[i]*He_1*de - rk04[i]*He_2*de - rk05[i]*He_2*de + rk06[i]*He_3*de;



    matrix_data[ j + 42] *= Tge[i];


    // He_3 by He_2
    colvals[j + 43] = i * nchem + 6 ;
    matrix_data[ j + 43 ] = k05[i]*de;




    // He_3 by He_3
    colvals[j + 44] = i * nchem + 7 ;
    matrix_data[ j + 44 ] = -k06[i]*de;




    // He_3 by de
    colvals[j + 45] = i * nchem + 8 ;
    matrix_data[ j + 45 ] = k05[i]*He_2 - k06[i]*He_3;




    // He_3 by ge
    colvals[j + 46] = i * nchem + 9 ;
    matrix_data[ j + 46 ] = rk05[i]*He_2*de - rk06[i]*He_3*de;



    matrix_data[ j + 46] *= Tge[i];


    // de by H2_2
    colvals[j + 47] = i * nchem + 1 ;
    matrix_data[ j + 47 ] = -k18[i]*de;




    // de by H_1
    colvals[j + 48] = i * nchem + 2 ;
    matrix_data[ j + 48 ] = k01[i]*de - k07[i]*de + k08[i]*H_m0 + k15[i]*H_m0;




    // de by H_2
    colvals[j + 49] = i * nchem + 3 ;
    matrix_data[ j + 49 ] = -k02[i]*de + k17[i]*H_m0;




    // de by H_m0
    colvals[j + 50] = i * nchem + 4 ;
    matrix_data[ j + 50 ] = k08[i]*H_1 + k14[i]*de + k15[i]*H_1 + k17[i]*H_2;




    // de by He_1
    colvals[j + 51] = i * nchem + 5 ;
    matrix_data[ j + 51 ] = k03[i]*de;




    // de by He_2
    colvals[j + 52] = i * nchem + 6 ;
    matrix_data[ j + 52 ] = -k04[i]*de + k05[i]*de;




    // de by He_3
    colvals[j + 53] = i * nchem + 7 ;
    matrix_data[ j + 53 ] = -k06[i]*de;




    // de by de
    colvals[j + 54] = i * nchem + 8 ;
    matrix_data[ j + 54 ] = k01[i]*H_1 - k02[i]*H_2 + k03[i]*He_1 - k04[i]*He_2 + k05[i]*He_2 - k06[i]*He_3 - k07[i]*H_1 + k14[i]*H_m0 - k18[i]*H2_2;




    // de by ge
    colvals[j + 55] = i * nchem + 9 ;
    matrix_data[ j + 55 ] = rk01[i]*H_1*de - rk02[i]*H_2*de + rk03[i]*He_1*de - rk04[i]*He_2*de + rk05[i]*He_2*de - rk06[i]*He_3*de - rk07[i]*H_1*de + rk08[i]*H_1*H_m0 + rk14[i]*H_m0*de + rk15[i]*H_1*H_m0 + rk17[i]*H_2*H_m0 - rk18[i]*H2_2*de;



    matrix_data[ j + 55] *= Tge[i];


    // ge by H2_1
    colvals[j + 56] = i * nchem + 0 ;
    matrix_data[ j + 56 ] = -H2_1*gloverabel08_gaH2[i]*pow(gloverabel08_h2lte[i], 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i], 2)) - 0.5*H_1*h2formation_h2mcool[i]*1.0/(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0) - 2.0158800000000001*cie_cooling_cieco[i]*mdensity - gloverabel08_h2lte[i]*h2_optical_depth_approx/(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0) + 0.5*h2formation_ncrd2[i]*h2formation_ncrn[i]*pow(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool[i] + pow(H_1, 3)*h2formation_h2mheat[i])/pow(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i], 2);


    matrix_data[j + 56] *= inv_mdensity;



    // ge by H_1
    colvals[j + 57] = i * nchem + 2 ;
    matrix_data[ j + 57 ] = -H2_1*gloverabel08_gaHI[i]*pow(gloverabel08_h2lte[i], 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i], 2)) - ceHI_ceHI[i]*de - ciHI_ciHI[i]*de + 0.5*h2formation_ncrd1[i]*h2formation_ncrn[i]*pow(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool[i] + pow(H_1, 3)*h2formation_h2mheat[i])/pow(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i], 2) + 0.5*(-H2_1*h2formation_h2mcool[i] + 3*pow(H_1, 2)*h2formation_h2mheat[i])*1.0/(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0);


    matrix_data[j + 57] *= inv_mdensity;



    // ge by H_2
    colvals[j + 58] = i * nchem + 3 ;
    matrix_data[ j + 58 ] = -H2_1*gloverabel08_gaHp[i]*pow(gloverabel08_h2lte[i], 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i], 2)) - brem_brem[i]*de - de*reHII_reHII[i];


    matrix_data[j + 58] *= inv_mdensity;



    // ge by He_1
    colvals[j + 59] = i * nchem + 5 ;
    matrix_data[ j + 59 ] = -H2_1*gloverabel08_gaHe[i]*pow(gloverabel08_h2lte[i], 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i], 2)) - ciHeI_ciHeI[i]*de;


    matrix_data[j + 59] *= inv_mdensity;



    // ge by He_2
    colvals[j + 60] = i * nchem + 6 ;
    matrix_data[ j + 60 ] = -brem_brem[i]*de - ceHeII_ceHeII[i]*de - ceHeI_ceHeI[i]*pow(de, 2) - ciHeII_ciHeII[i]*de - ciHeIS_ciHeIS[i]*pow(de, 2) - de*reHeII1_reHeII1[i] - de*reHeII2_reHeII2[i];


    matrix_data[j + 60] *= inv_mdensity;



    // ge by He_3
    colvals[j + 61] = i * nchem + 7 ;
    matrix_data[ j + 61 ] = -4.0*brem_brem[i]*de - de*reHeIII_reHeIII[i];


    matrix_data[j + 61] *= inv_mdensity;



    // ge by de
    colvals[j + 62] = i * nchem + 8 ;
    matrix_data[ j + 62 ] = -H2_1*gloverabel08_gael[i]*pow(gloverabel08_h2lte[i], 2)*h2_optical_depth_approx/(pow(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0, 2)*pow(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i], 2)) - H_1*ceHI_ceHI[i] - H_1*ciHI_ciHI[i] - H_2*reHII_reHII[i] - He_1*ciHeI_ciHeI[i] - He_2*ceHeII_ceHeII[i] - 2*He_2*ceHeI_ceHeI[i]*de - He_2*ciHeII_ciHeII[i] - 2*He_2*ciHeIS_ciHeIS[i]*de - He_2*reHeII1_reHeII1[i] - He_2*reHeII2_reHeII2[i] - He_3*reHeIII_reHeIII[i] - brem_brem[i]*(H_2 + He_2 + 4.0*He_3) - compton_comp_[i]*pow(z + 1.0, 4)*(T - 2.73*z - 2.73);


    matrix_data[j + 62] *= inv_mdensity;



    // ge by ge
    colvals[j + 63] = i * nchem + 9 ;
    matrix_data[ j + 63 ] = -2.0158800000000001*H2_1*cie_cooling_cieco[i]*cie_optical_depth_approx*mdensity - H2_1*cie_optical_depth_approx*gloverabel08_h2lte[i]*h2_optical_depth_approx/(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0) - H_1*ceHI_ceHI[i]*cie_optical_depth_approx*de - H_1*ciHI_ciHI[i]*cie_optical_depth_approx*de - H_2*cie_optical_depth_approx*de*reHII_reHII[i] - He_1*ciHeI_ciHeI[i]*cie_optical_depth_approx*de - He_2*ceHeII_ceHeII[i]*cie_optical_depth_approx*de - He_2*ceHeI_ceHeI[i]*cie_optical_depth_approx*pow(de, 2) - He_2*ciHeII_ciHeII[i]*cie_optical_depth_approx*de - He_2*ciHeIS_ciHeIS[i]*cie_optical_depth_approx*pow(de, 2) - He_2*cie_optical_depth_approx*de*reHeII1_reHeII1[i] - He_2*cie_optical_depth_approx*de*reHeII2_reHeII2[i] - He_3*cie_optical_depth_approx*de*reHeIII_reHeIII[i] - brem_brem[i]*cie_optical_depth_approx*de*(H_2 + He_2 + 4.0*He_3) - cie_optical_depth_approx*compton_comp_[i]*de*pow(z + 1.0, 4)*(T - 2.73*z - 2.73) + 0.5*1.0/(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0)*(-H2_1*H_1*h2formation_h2mcool[i] + pow(H_1, 3)*h2formation_h2mheat[i]);

    // ad-hoc extra term of f_ge by ge
    // considering ONLY the h2formation/ and continuum cooling
    matrix_data[ j + 63] = -H2_1*gloverabel08_h2lte[i]*h2_optical_depth_approx*(-gloverabel08_h2lte[i]*(-H2_1*rgloverabel08_gaH2[i] - H_1*rgloverabel08_gaHI[i] - H_2*rgloverabel08_gaHp[i] - He_1*rgloverabel08_gaHe[i] - de*rgloverabel08_gael[i])/pow(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i], 2) - rgloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]))/pow(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0, 2) - H2_1*h2_optical_depth_approx*rgloverabel08_h2lte[i]/(gloverabel08_h2lte[i]/(H2_1*gloverabel08_gaH2[i] + H_1*gloverabel08_gaHI[i] + H_2*gloverabel08_gaHp[i] + He_1*gloverabel08_gaHe[i] + de*gloverabel08_gael[i]) + 1.0) + 0.5*pow(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0, -2.0)*(-H2_1*H_1*h2formation_h2mcool[i] + pow(H_1, 3)*h2formation_h2mheat[i])*(-1.0*h2formation_ncrn[i]*(-H2_1*rh2formation_ncrd2[i] - H_1*rh2formation_ncrd1[i])/pow(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i], 2) - 1.0*rh2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i])) + 0.5*1.0/(h2formation_ncrn[i]/(H2_1*h2formation_ncrd2[i] + H_1*h2formation_ncrd1[i]) + 1.0)*(-H2_1*H_1*rh2formation_h2mcool[i] + pow(H_1, 3)*rh2formation_h2mheat[i]);


    matrix_data[j + 63] *= inv_mdensity;


    matrix_data[ j + 63] *= Tge[i];




    rowptrs[ i * nchem +  0] = i * NSPARSE + 0;

    rowptrs[ i * nchem +  1] = i * NSPARSE + 7;

    rowptrs[ i * nchem +  2] = i * NSPARSE + 14;

    rowptrs[ i * nchem +  3] = i * NSPARSE + 21;

    rowptrs[ i * nchem +  4] = i * NSPARSE + 28;

    rowptrs[ i * nchem +  5] = i * NSPARSE + 34;

    rowptrs[ i * nchem +  6] = i * NSPARSE + 38;

    rowptrs[ i * nchem +  7] = i * NSPARSE + 43;

    rowptrs[ i * nchem +  8] = i * NSPARSE + 47;

    rowptrs[ i * nchem +  9] = i * NSPARSE + 56;


    j = i * nchem;

    inv_scale1 = inv_scale[ j + 0 ];
    scale2     = scale    [ j + 0 ];
    matrix_data[ i * NSPARSE + 0]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 0 ];
    scale2     = scale    [ j + 1 ];
    matrix_data[ i * NSPARSE + 1]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 0 ];
    scale2     = scale    [ j + 2 ];
    matrix_data[ i * NSPARSE + 2]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 0 ];
    scale2     = scale    [ j + 3 ];
    matrix_data[ i * NSPARSE + 3]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 0 ];
    scale2     = scale    [ j + 4 ];
    matrix_data[ i * NSPARSE + 4]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 0 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 5]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 0 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 6]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 1 ];
    scale2     = scale    [ j + 0 ];
    matrix_data[ i * NSPARSE + 7]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 1 ];
    scale2     = scale    [ j + 1 ];
    matrix_data[ i * NSPARSE + 8]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 1 ];
    scale2     = scale    [ j + 2 ];
    matrix_data[ i * NSPARSE + 9]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 1 ];
    scale2     = scale    [ j + 3 ];
    matrix_data[ i * NSPARSE + 10]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 1 ];
    scale2     = scale    [ j + 4 ];
    matrix_data[ i * NSPARSE + 11]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 1 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 12]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 1 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 13]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 2 ];
    scale2     = scale    [ j + 0 ];
    matrix_data[ i * NSPARSE + 14]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 2 ];
    scale2     = scale    [ j + 1 ];
    matrix_data[ i * NSPARSE + 15]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 2 ];
    scale2     = scale    [ j + 2 ];
    matrix_data[ i * NSPARSE + 16]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 2 ];
    scale2     = scale    [ j + 3 ];
    matrix_data[ i * NSPARSE + 17]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 2 ];
    scale2     = scale    [ j + 4 ];
    matrix_data[ i * NSPARSE + 18]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 2 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 19]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 2 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 20]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 3 ];
    scale2     = scale    [ j + 0 ];
    matrix_data[ i * NSPARSE + 21]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 3 ];
    scale2     = scale    [ j + 1 ];
    matrix_data[ i * NSPARSE + 22]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 3 ];
    scale2     = scale    [ j + 2 ];
    matrix_data[ i * NSPARSE + 23]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 3 ];
    scale2     = scale    [ j + 3 ];
    matrix_data[ i * NSPARSE + 24]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 3 ];
    scale2     = scale    [ j + 4 ];
    matrix_data[ i * NSPARSE + 25]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 3 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 26]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 3 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 27]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 4 ];
    scale2     = scale    [ j + 1 ];
    matrix_data[ i * NSPARSE + 28]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 4 ];
    scale2     = scale    [ j + 2 ];
    matrix_data[ i * NSPARSE + 29]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 4 ];
    scale2     = scale    [ j + 3 ];
    matrix_data[ i * NSPARSE + 30]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 4 ];
    scale2     = scale    [ j + 4 ];
    matrix_data[ i * NSPARSE + 31]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 4 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 32]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 4 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 33]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 5 ];
    scale2     = scale    [ j + 5 ];
    matrix_data[ i * NSPARSE + 34]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 5 ];
    scale2     = scale    [ j + 6 ];
    matrix_data[ i * NSPARSE + 35]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 5 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 36]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 5 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 37]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 6 ];
    scale2     = scale    [ j + 5 ];
    matrix_data[ i * NSPARSE + 38]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 6 ];
    scale2     = scale    [ j + 6 ];
    matrix_data[ i * NSPARSE + 39]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 6 ];
    scale2     = scale    [ j + 7 ];
    matrix_data[ i * NSPARSE + 40]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 6 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 41]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 6 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 42]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 7 ];
    scale2     = scale    [ j + 6 ];
    matrix_data[ i * NSPARSE + 43]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 7 ];
    scale2     = scale    [ j + 7 ];
    matrix_data[ i * NSPARSE + 44]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 7 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 45]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 7 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 46]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 1 ];
    matrix_data[ i * NSPARSE + 47]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 2 ];
    matrix_data[ i * NSPARSE + 48]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 3 ];
    matrix_data[ i * NSPARSE + 49]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 4 ];
    matrix_data[ i * NSPARSE + 50]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 5 ];
    matrix_data[ i * NSPARSE + 51]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 6 ];
    matrix_data[ i * NSPARSE + 52]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 7 ];
    matrix_data[ i * NSPARSE + 53]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 54]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 8 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 55]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 0 ];
    matrix_data[ i * NSPARSE + 56]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 2 ];
    matrix_data[ i * NSPARSE + 57]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 3 ];
    matrix_data[ i * NSPARSE + 58]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 5 ];
    matrix_data[ i * NSPARSE + 59]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 6 ];
    matrix_data[ i * NSPARSE + 60]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 7 ];
    matrix_data[ i * NSPARSE + 61]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 8 ];
    matrix_data[ i * NSPARSE + 62]  *= inv_scale1*scale2;

    inv_scale1 = inv_scale[ j + 9 ];
    scale2     = scale    [ j + 9 ];
    matrix_data[ i * NSPARSE + 63]  *= inv_scale1*scale2;



  }

  rowptrs[ i * nchem ] = i * NSPARSE ;
  return 0;
}




void setting_up_extra_variables( cvklu_data * data, double * input, int nstrip ){

  int i, j;
  double mh = 1.67e-24;
  for ( i = 0; i < nstrip; i++){
    data->mdensity[i] = 0;
    j = i * 10;

    // species: H2_1
    data->mdensity[i] += input[j] * 2.0;

    j ++;


    // species: H2_2
    data->mdensity[i] += input[j] * 2.0;

    j ++;


    // species: H_1
    data->mdensity[i] += input[j] * 1.00794;

    j ++;


    // species: H_2
    data->mdensity[i] += input[j] * 1.00794;

    j ++;


    // species: H_m0
    data->mdensity[i] += input[j] * 1.00794;

    j ++;


    // species: He_1
    data->mdensity[i] += input[j] * 4.002602;

    j ++;


    // species: He_2
    data->mdensity[i] += input[j] * 4.002602;

    j ++;


    // species: He_3
    data->mdensity[i] += input[j] * 4.002602;

    j ++;


    j ++;


    j ++;

    data->mdensity[i] *= mh;
    data->inv_mdensity[i] = 1.0 / data->mdensity[i];
  }


  double mdensity, tau;
  for ( i = 0; i < nstrip; i++){

    mdensity = data->mdensity[i];
    tau      = pow( (mdensity / 3.3e-8 ), 2.8);
    tau      = fmax( tau, 1.0e-5 );
    data->cie_optical_depth_approx[i] =
      fmin( 1.0, (1.0 - exp(-tau) ) / tau );
  }




  for ( i = 0; i < nstrip; i++ ){
    mdensity = data->mdensity[i];
    data->h2_optical_depth_approx[i] =  fmin( 1.0, pow( (mdensity / (1.34e-14) )  , -0.45) );
  }



}
