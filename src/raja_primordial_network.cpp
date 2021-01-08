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

  cvklu_data *data = (cvklu_data *) malloc(sizeof(cvklu_data));

  // point the module to look for cvklu_tables.h5
  data->dengo_data_file = FileLocation;

  // allocate reaction rate structures
  data->cell_data = (cell_rate_data *) malloc(ncells*sizeof(cell_rate_data));

  // allocate scaling arrays
  data->scale = (double *) malloc(NSPECIES*ncells*sizeof(double));
  data->inv_scale = (double *) malloc(NSPECIES*ncells*sizeof(double));
  data->current_z = 0.0;

  // Number of cells to be solved in a batch
  data->nstrip = ncells;

  // initialize temperature so it wont crash
  //  for ( i = 0; i < ncells; i++ ) {
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,ncells), [=] RAJA_DEVICE (int i) {
    (data->cell_data[i]).Ts = 1000.0;
  //}
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

  free(rxdata->cell_data);
  free(rxdata->scale);
  free(rxdata->inv_scale);
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

  // Allocate the correct number of rate tables
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

  // Read the rate tables
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

  // Read the gamma tables
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



int cvklu_calculate_temperature(cvklu_data *data, double *input, cell_rate_data &rdata)
{

  // Define some constants
  const double kb = 1.3806504e-16; // Boltzmann constant [erg/K]
  const double mh = 1.67e-24;
  const double gamma = 5.e0/3.e0;
  const double _gamma_m1 = 1.0 / (gamma - 1);
  const double gammaH2 = 7.e0/5.e0;
  const int MAX_T_ITERATION = 100;

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
  double density = 2.0*H2_1 + 2.0*H2_2 + 1.0079400000000001*H_1 + 1.0079400000000001*H_2 + 1.0079400000000001*H_m0 + 4.0026020000000004*He_1 + 4.0026020000000004*He_2 + 4.0026020000000004*He_3;

  // Initiate the "guess" temperature
  double T    = rdata.Ts;
  double Tnew = T*1.1;
  double Tdiff = Tnew - T;
  double dge_dT;
  int count = 0;

  // We do Newton's Iteration to calculate the temperature
  // Since gammaH2 is dependent on the temperature too!
  //        while ( Tdiff/ Tnew > 0.001 ){
  for (int j=0; j<10; j++){

    T = rdata.Ts;
    cvklu_interpolate_gamma(data, rdata);

    double gammaH2_1 = rdata.gammaH2_1;
    double dgammaH2_1_dT = rdata.dgammaH2_1_dT;
    double _gammaH2_1_m1 = 1.0 / (gammaH2_1 - 1.0);

    double gammaH2_2 = rdata.gammaH2_2;
    double dgammaH2_2_dT = rdata.dgammaH2_2_dT;
    double _gammaH2_2_m1 = 1.0 / (gammaH2_2 - 1.0);

    // update gammaH2
    // The derivatives of  sum (nkT/(gamma - 1)/mh/density) - ge
    // This is the function we want to minimize
    // which should only be dependent on the first part
    dge_dT = T*kb*(-H2_1*_gammaH2_1_m1*_gammaH2_1_m1*dgammaH2_1_dT - H2_2*_gammaH2_2_m1*_gammaH2_2_m1*dgammaH2_2_dT)/(density*mh) + kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh);

    //This is the change in ge for each iteration
    double dge = T*kb*(H2_1*_gammaH2_1_m1 + H2_2*_gammaH2_2_m1 + H_1*_gamma_m1 + H_2*_gamma_m1 + H_m0*_gamma_m1 + He_1*_gamma_m1 + He_2*_gamma_m1 + He_3*_gamma_m1 + _gamma_m1*de)/(density*mh) - ge;

    Tnew = T - dge/dge_dT;
    rdata.Ts = Tnew;

    Tdiff = fabs(T - Tnew);
    // fprintf(stderr, "T: %0.5g ; Tnew: %0.5g; dge_dT: %.5g, dge: %.5g, ge: %.5g \n", T,Tnew, dge_dT, dge, ge);
    // count += 1;
    // if (count > MAX_T_ITERATION){
    //     fprintf(stderr, "T failed to converge \n");
    //     return 1;
    // }
  } // while loop

  rdata.Ts = Tnew;

  if (rdata.Ts < data->bounds[0]) {
    rdata.Ts = data->bounds[0];
  } else if (rdata.Ts > data->bounds[1]) {
    rdata.Ts = data->bounds[1];
  }
  rdata.dTs_ge = 1.0 / dge_dT;

  return 0;

}



void cvklu_interpolate_rates(cvklu_data *data, cell_rate_data &rdata)
{
  int bin_id, zbin_id;
  double lb, t1, t2;
  double Tdef, dT, invTs, Tfactor;

  lb = log(data->bounds[0]);

  bin_id = (int) (data->idbin * (log(rdata.Ts) - lb));
  if (bin_id <= 0) {
    bin_id = 0;
  } else if (bin_id >= data->nbins) {
    bin_id = data->nbins - 1;
  }
  t1 = (lb + (bin_id    ) * data->dbin);
  t2 = (lb + (bin_id + 1) * data->dbin);
  Tdef = (log(rdata.Ts) - t1)/(t2 - t1);
  dT = (t2 - t1);
  invTs = 1.0 / rdata.Ts;
  Tfactor = invTs/dT;

  rdata.rs_k01 = data->r_k01[bin_id] +
    Tdef * (data->r_k01[bin_id+1] - data->r_k01[bin_id]);
  rdata.drs_k01 = (data->r_k01[bin_id+1] - data->r_k01[bin_id])*Tfactor;

  rdata.rs_k02 = data->r_k02[bin_id] +
    Tdef * (data->r_k02[bin_id+1] - data->r_k02[bin_id]);
  rdata.drs_k02 = (data->r_k02[bin_id+1] - data->r_k02[bin_id])*Tfactor;

  rdata.rs_k03 = data->r_k03[bin_id] +
    Tdef * (data->r_k03[bin_id+1] - data->r_k03[bin_id]);
  rdata.drs_k03 = (data->r_k03[bin_id+1] - data->r_k03[bin_id])*Tfactor;

  rdata.rs_k04 = data->r_k04[bin_id] +
    Tdef * (data->r_k04[bin_id+1] - data->r_k04[bin_id]);
  rdata.drs_k04 = (data->r_k04[bin_id+1] - data->r_k04[bin_id])*Tfactor;

  rdata.rs_k05 = data->r_k05[bin_id] +
    Tdef * (data->r_k05[bin_id+1] - data->r_k05[bin_id]);
  rdata.drs_k05 = (data->r_k05[bin_id+1] - data->r_k05[bin_id])*Tfactor;

  rdata.rs_k06 = data->r_k06[bin_id] +
    Tdef * (data->r_k06[bin_id+1] - data->r_k06[bin_id]);
  rdata.drs_k06 = (data->r_k06[bin_id+1] - data->r_k06[bin_id])*Tfactor;

  rdata.rs_k07 = data->r_k07[bin_id] +
    Tdef * (data->r_k07[bin_id+1] - data->r_k07[bin_id]);
  rdata.drs_k07 = (data->r_k07[bin_id+1] - data->r_k07[bin_id])*Tfactor;

  rdata.rs_k08 = data->r_k08[bin_id] +
    Tdef * (data->r_k08[bin_id+1] - data->r_k08[bin_id]);
  rdata.drs_k08 = (data->r_k08[bin_id+1] - data->r_k08[bin_id])*Tfactor;

  rdata.rs_k09 = data->r_k09[bin_id] +
    Tdef * (data->r_k09[bin_id+1] - data->r_k09[bin_id]);
  rdata.drs_k09 = (data->r_k09[bin_id+1] - data->r_k09[bin_id])*Tfactor;

  rdata.rs_k10 = data->r_k10[bin_id] +
    Tdef * (data->r_k10[bin_id+1] - data->r_k10[bin_id]);
  rdata.drs_k10 = (data->r_k10[bin_id+1] - data->r_k10[bin_id])*Tfactor;

  rdata.rs_k11 = data->r_k11[bin_id] +
    Tdef * (data->r_k11[bin_id+1] - data->r_k11[bin_id]);
  rdata.drs_k11 = (data->r_k11[bin_id+1] - data->r_k11[bin_id])*Tfactor;

  rdata.rs_k12 = data->r_k12[bin_id] +
    Tdef * (data->r_k12[bin_id+1] - data->r_k12[bin_id]);
  rdata.drs_k12 = (data->r_k12[bin_id+1] - data->r_k12[bin_id])*Tfactor;

  rdata.rs_k13 = data->r_k13[bin_id] +
    Tdef * (data->r_k13[bin_id+1] - data->r_k13[bin_id]);
  rdata.drs_k13 = (data->r_k13[bin_id+1] - data->r_k13[bin_id])*Tfactor;

  rdata.rs_k14 = data->r_k14[bin_id] +
    Tdef * (data->r_k14[bin_id+1] - data->r_k14[bin_id]);
  rdata.drs_k14 = (data->r_k14[bin_id+1] - data->r_k14[bin_id])*Tfactor;

  rdata.rs_k15 = data->r_k15[bin_id] +
    Tdef * (data->r_k15[bin_id+1] - data->r_k15[bin_id]);
  rdata.drs_k15 = (data->r_k15[bin_id+1] - data->r_k15[bin_id])*Tfactor;

  rdata.rs_k16 = data->r_k16[bin_id] +
    Tdef * (data->r_k16[bin_id+1] - data->r_k16[bin_id]);
  rdata.drs_k16 = (data->r_k16[bin_id+1] - data->r_k16[bin_id])*Tfactor;

  rdata.rs_k17 = data->r_k17[bin_id] +
    Tdef * (data->r_k17[bin_id+1] - data->r_k17[bin_id]);
  rdata.drs_k17 = (data->r_k17[bin_id+1] - data->r_k17[bin_id])*Tfactor;

  rdata.rs_k18 = data->r_k18[bin_id] +
    Tdef * (data->r_k18[bin_id+1] - data->r_k18[bin_id]);
  rdata.drs_k18 = (data->r_k18[bin_id+1] - data->r_k18[bin_id])*Tfactor;

  rdata.rs_k19 = data->r_k19[bin_id] +
    Tdef * (data->r_k19[bin_id+1] - data->r_k19[bin_id]);
  rdata.drs_k19 = (data->r_k19[bin_id+1] - data->r_k19[bin_id])*Tfactor;

  rdata.rs_k21 = data->r_k21[bin_id] +
    Tdef * (data->r_k21[bin_id+1] - data->r_k21[bin_id]);
  rdata.drs_k21 = (data->r_k21[bin_id+1] - data->r_k21[bin_id])*Tfactor;

  rdata.rs_k22 = data->r_k22[bin_id] +
    Tdef * (data->r_k22[bin_id+1] - data->r_k22[bin_id]);
  rdata.drs_k22 = (data->r_k22[bin_id+1] - data->r_k22[bin_id])*Tfactor;

  rdata.cs_brem_brem = data->c_brem_brem[bin_id] +
    Tdef * (data->c_brem_brem[bin_id+1] - data->c_brem_brem[bin_id]);
  rdata.dcs_brem_brem = (data->c_brem_brem[bin_id+1] - data->c_brem_brem[bin_id])*Tfactor;

  rdata.cs_ceHeI_ceHeI = data->c_ceHeI_ceHeI[bin_id] +
    Tdef * (data->c_ceHeI_ceHeI[bin_id+1] - data->c_ceHeI_ceHeI[bin_id]);
  rdata.dcs_ceHeI_ceHeI = (data->c_ceHeI_ceHeI[bin_id+1] - data->c_ceHeI_ceHeI[bin_id])*Tfactor;

  rdata.cs_ceHeII_ceHeII = data->c_ceHeII_ceHeII[bin_id] +
    Tdef * (data->c_ceHeII_ceHeII[bin_id+1] - data->c_ceHeII_ceHeII[bin_id]);
  rdata.dcs_ceHeII_ceHeII = (data->c_ceHeII_ceHeII[bin_id+1] - data->c_ceHeII_ceHeII[bin_id])*Tfactor;

  rdata.cs_ceHI_ceHI = data->c_ceHI_ceHI[bin_id] +
    Tdef * (data->c_ceHI_ceHI[bin_id+1] - data->c_ceHI_ceHI[bin_id]);
  rdata.dcs_ceHI_ceHI = (data->c_ceHI_ceHI[bin_id+1] - data->c_ceHI_ceHI[bin_id])*Tfactor;

  rdata.cs_cie_cooling_cieco = data->c_cie_cooling_cieco[bin_id] +
    Tdef * (data->c_cie_cooling_cieco[bin_id+1] - data->c_cie_cooling_cieco[bin_id]);
  rdata.dcs_cie_cooling_cieco = (data->c_cie_cooling_cieco[bin_id+1] - data->c_cie_cooling_cieco[bin_id])*Tfactor;

  rdata.cs_ciHeI_ciHeI = data->c_ciHeI_ciHeI[bin_id] +
    Tdef * (data->c_ciHeI_ciHeI[bin_id+1] - data->c_ciHeI_ciHeI[bin_id]);
  rdata.dcs_ciHeI_ciHeI = (data->c_ciHeI_ciHeI[bin_id+1] - data->c_ciHeI_ciHeI[bin_id])*Tfactor;

  rdata.cs_ciHeII_ciHeII = data->c_ciHeII_ciHeII[bin_id] +
    Tdef * (data->c_ciHeII_ciHeII[bin_id+1] - data->c_ciHeII_ciHeII[bin_id]);
  rdata.dcs_ciHeII_ciHeII = (data->c_ciHeII_ciHeII[bin_id+1] - data->c_ciHeII_ciHeII[bin_id])*Tfactor;

  rdata.cs_ciHeIS_ciHeIS = data->c_ciHeIS_ciHeIS[bin_id] +
    Tdef * (data->c_ciHeIS_ciHeIS[bin_id+1] - data->c_ciHeIS_ciHeIS[bin_id]);
  rdata.dcs_ciHeIS_ciHeIS = (data->c_ciHeIS_ciHeIS[bin_id+1] - data->c_ciHeIS_ciHeIS[bin_id])*Tfactor;

  rdata.cs_ciHI_ciHI = data->c_ciHI_ciHI[bin_id] +
    Tdef * (data->c_ciHI_ciHI[bin_id+1] - data->c_ciHI_ciHI[bin_id]);
  rdata.dcs_ciHI_ciHI = (data->c_ciHI_ciHI[bin_id+1] - data->c_ciHI_ciHI[bin_id])*Tfactor;

  rdata.cs_compton_comp_ = data->c_compton_comp_[bin_id] +
    Tdef * (data->c_compton_comp_[bin_id+1] - data->c_compton_comp_[bin_id]);
  rdata.dcs_compton_comp_ = (data->c_compton_comp_[bin_id+1] - data->c_compton_comp_[bin_id])*Tfactor;

  rdata.cs_gloverabel08_gael = data->c_gloverabel08_gael[bin_id] +
    Tdef * (data->c_gloverabel08_gael[bin_id+1] - data->c_gloverabel08_gael[bin_id]);
  rdata.dcs_gloverabel08_gael = (data->c_gloverabel08_gael[bin_id+1] - data->c_gloverabel08_gael[bin_id])*Tfactor;

  rdata.cs_gloverabel08_gaH2 = data->c_gloverabel08_gaH2[bin_id] +
    Tdef * (data->c_gloverabel08_gaH2[bin_id+1] - data->c_gloverabel08_gaH2[bin_id]);
  rdata.dcs_gloverabel08_gaH2 = (data->c_gloverabel08_gaH2[bin_id+1] - data->c_gloverabel08_gaH2[bin_id])*Tfactor;

  rdata.cs_gloverabel08_gaHe = data->c_gloverabel08_gaHe[bin_id] +
    Tdef * (data->c_gloverabel08_gaHe[bin_id+1] - data->c_gloverabel08_gaHe[bin_id]);
  rdata.dcs_gloverabel08_gaHe = (data->c_gloverabel08_gaHe[bin_id+1] - data->c_gloverabel08_gaHe[bin_id])*Tfactor;

  rdata.cs_gloverabel08_gaHI = data->c_gloverabel08_gaHI[bin_id] +
    Tdef * (data->c_gloverabel08_gaHI[bin_id+1] - data->c_gloverabel08_gaHI[bin_id]);
  rdata.dcs_gloverabel08_gaHI = (data->c_gloverabel08_gaHI[bin_id+1] - data->c_gloverabel08_gaHI[bin_id])*Tfactor;

  rdata.cs_gloverabel08_gaHp = data->c_gloverabel08_gaHp[bin_id] +
    Tdef * (data->c_gloverabel08_gaHp[bin_id+1] - data->c_gloverabel08_gaHp[bin_id]);
  rdata.dcs_gloverabel08_gaHp = (data->c_gloverabel08_gaHp[bin_id+1] - data->c_gloverabel08_gaHp[bin_id])*Tfactor;

  rdata.cs_gloverabel08_h2lte = data->c_gloverabel08_h2lte[bin_id] +
    Tdef * (data->c_gloverabel08_h2lte[bin_id+1] - data->c_gloverabel08_h2lte[bin_id]);
  rdata.dcs_gloverabel08_h2lte = (data->c_gloverabel08_h2lte[bin_id+1] - data->c_gloverabel08_h2lte[bin_id])*Tfactor;

  rdata.cs_h2formation_h2mcool = data->c_h2formation_h2mcool[bin_id] +
    Tdef * (data->c_h2formation_h2mcool[bin_id+1] - data->c_h2formation_h2mcool[bin_id]);
  rdata.dcs_h2formation_h2mcool = (data->c_h2formation_h2mcool[bin_id+1] - data->c_h2formation_h2mcool[bin_id])*Tfactor;

  rdata.cs_h2formation_h2mheat = data->c_h2formation_h2mheat[bin_id] +
    Tdef * (data->c_h2formation_h2mheat[bin_id+1] - data->c_h2formation_h2mheat[bin_id]);
  rdata.dcs_h2formation_h2mheat = (data->c_h2formation_h2mheat[bin_id+1] - data->c_h2formation_h2mheat[bin_id])*Tfactor;

  rdata.cs_h2formation_ncrd1 = data->c_h2formation_ncrd1[bin_id] +
    Tdef * (data->c_h2formation_ncrd1[bin_id+1] - data->c_h2formation_ncrd1[bin_id]);
  rdata.dcs_h2formation_ncrd1 = (data->c_h2formation_ncrd1[bin_id+1] - data->c_h2formation_ncrd1[bin_id])*Tfactor;

  rdata.cs_h2formation_ncrd2 = data->c_h2formation_ncrd2[bin_id] +
    Tdef * (data->c_h2formation_ncrd2[bin_id+1] - data->c_h2formation_ncrd2[bin_id]);
  rdata.dcs_h2formation_ncrd2 = (data->c_h2formation_ncrd2[bin_id+1] - data->c_h2formation_ncrd2[bin_id])*Tfactor;

  rdata.cs_h2formation_ncrn = data->c_h2formation_ncrn[bin_id] +
    Tdef * (data->c_h2formation_ncrn[bin_id+1] - data->c_h2formation_ncrn[bin_id]);
  rdata.dcs_h2formation_ncrn = (data->c_h2formation_ncrn[bin_id+1] - data->c_h2formation_ncrn[bin_id])*Tfactor;

  rdata.cs_reHeII1_reHeII1 = data->c_reHeII1_reHeII1[bin_id] +
    Tdef * (data->c_reHeII1_reHeII1[bin_id+1] - data->c_reHeII1_reHeII1[bin_id]);
  rdata.dcs_reHeII1_reHeII1 = (data->c_reHeII1_reHeII1[bin_id+1] - data->c_reHeII1_reHeII1[bin_id])*Tfactor;

  rdata.cs_reHeII2_reHeII2 = data->c_reHeII2_reHeII2[bin_id] +
    Tdef * (data->c_reHeII2_reHeII2[bin_id+1] - data->c_reHeII2_reHeII2[bin_id]);
  rdata.dcs_reHeII2_reHeII2 = (data->c_reHeII2_reHeII2[bin_id+1] - data->c_reHeII2_reHeII2[bin_id])*Tfactor;

  rdata.cs_reHeIII_reHeIII = data->c_reHeIII_reHeIII[bin_id] +
    Tdef * (data->c_reHeIII_reHeIII[bin_id+1] - data->c_reHeIII_reHeIII[bin_id]);
  rdata.dcs_reHeIII_reHeIII = (data->c_reHeIII_reHeIII[bin_id+1] - data->c_reHeIII_reHeIII[bin_id])*Tfactor;

  rdata.cs_reHII_reHII = data->c_reHII_reHII[bin_id] +
    Tdef * (data->c_reHII_reHII[bin_id+1] - data->c_reHII_reHII[bin_id]);
  rdata.dcs_reHII_reHII = (data->c_reHII_reHII[bin_id+1] - data->c_reHII_reHII[bin_id])*Tfactor;


}



void cvklu_interpolate_gamma(cvklu_data *data, cell_rate_data &rdata)
{

  int bin_id, zbin_id;
  double lb, t1, t2, Tdef;

  lb = log(data->bounds[0]);
  bin_id = (int) (data->idbin * (log(rdata.Ts) - lb));
  if (bin_id <= 0) {
    bin_id = 0;
  } else if (bin_id >= data->nbins) {
    bin_id = data->nbins - 1;
  }
  t1 = (lb + (bin_id    ) * data->dbin);
  t2 = (lb + (bin_id + 1) * data->dbin);
  Tdef = (log(rdata.Ts) - t1)/(t2 - t1);


  rdata.gammaH2_2 = data->g_gammaH2_2[bin_id] +
    Tdef * (data->g_gammaH2_2[bin_id+1] - data->g_gammaH2_2[bin_id]);

  rdata.dgammaH2_2_dT = data->g_dgammaH2_2_dT[bin_id] +
    Tdef * (data->g_dgammaH2_2_dT[bin_id+1]
                     - data->g_dgammaH2_2_dT[bin_id]);


  rdata.gammaH2_1 = data->g_gammaH2_1[bin_id] +
    Tdef * (data->g_gammaH2_1[bin_id+1] - data->g_gammaH2_1[bin_id]);

  rdata.dgammaH2_1_dT = data->g_dgammaH2_1_dT[bin_id] +
    Tdef * (data->g_dgammaH2_1_dT[bin_id+1]
                     - data->g_dgammaH2_1_dT[bin_id]);

}




int calculate_rhs_cvklu(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  cvklu_data *data = (cvklu_data* ) user_data;
  const double mh   = 1.67e-24;
  double *scale     = data->scale;
  double *inv_scale = data->inv_scale;
  double *ydata     = N_VGetArrayPointer(y);
  double *ydotdata  = N_VGetArrayPointer(ydot);

  //  for (int i = 0; i < data->nstrip; i++ ){
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,data->nstrip), [=] RAJA_DEVICE (int i) {

    double y_arr[NSPECIES];
    int j = i * NSPECIES;
    double H2_1 = y_arr[0] = ydata[j]*scale[j];
    j++;
    double H2_2 = y_arr[1] = ydata[j]*scale[j];
    j++;
    double H_1 = y_arr[2] = ydata[j]*scale[j];
    j++;
    double H_2 = y_arr[3] = ydata[j]*scale[j];
    j++;
    double H_m0 = y_arr[4] = ydata[j]*scale[j];
    j++;
    double He_1 = y_arr[5] = ydata[j]*scale[j];
    j++;
    double He_2 = y_arr[6] = ydata[j]*scale[j];
    j++;
    double He_3 = y_arr[7] = ydata[j]*scale[j];
    j++;
    double de = y_arr[8] = ydata[j]*scale[j];
    j++;
    double ge = y_arr[9] = ydata[j]*scale[j];
    j++;

    // Calculate temperature in this cell
    int flag = cvklu_calculate_temperature(data, y_arr, data->cell_data[i]);
    if (flag > 0) {
      return 1;   // return recoverable failure if temperature failed to converged
    }

    // Calculate reaction rates in this cell
    cvklu_interpolate_rates(data, data->cell_data[i]);

    // Set up some temporaries
    const double T = (data->cell_data[i]).Ts;
    const double z = data->current_z;
    const double mdensity = (data->cell_data[i]).mdensity;
    const double inv_mdensity = (data->cell_data[i]).inv_mdensity;
    const double k01 = (data->cell_data[i]).rs_k01;
    const double k02 = (data->cell_data[i]).rs_k02;
    const double k03 = (data->cell_data[i]).rs_k03;
    const double k04 = (data->cell_data[i]).rs_k04;
    const double k05 = (data->cell_data[i]).rs_k05;
    const double k06 = (data->cell_data[i]).rs_k06;
    const double k07 = (data->cell_data[i]).rs_k07;
    const double k08 = (data->cell_data[i]).rs_k08;
    const double k09 = (data->cell_data[i]).rs_k09;
    const double k10 = (data->cell_data[i]).rs_k10;
    const double k11 = (data->cell_data[i]).rs_k11;
    const double k12 = (data->cell_data[i]).rs_k12;
    const double k13 = (data->cell_data[i]).rs_k13;
    const double k14 = (data->cell_data[i]).rs_k14;
    const double k15 = (data->cell_data[i]).rs_k15;
    const double k16 = (data->cell_data[i]).rs_k16;
    const double k17 = (data->cell_data[i]).rs_k17;
    const double k18 = (data->cell_data[i]).rs_k18;
    const double k19 = (data->cell_data[i]).rs_k19;
    const double k21 = (data->cell_data[i]).rs_k21;
    const double k22 = (data->cell_data[i]).rs_k22;
    const double brem_brem = (data->cell_data[i]).cs_brem_brem;
    const double ceHeI_ceHeI = (data->cell_data[i]).cs_ceHeI_ceHeI;
    const double ceHeII_ceHeII = (data->cell_data[i]).cs_ceHeII_ceHeII;
    const double ceHI_ceHI = (data->cell_data[i]).cs_ceHI_ceHI;
    const double cie_cooling_cieco = (data->cell_data[i]).cs_cie_cooling_cieco;
    const double ciHeI_ciHeI = (data->cell_data[i]).cs_ciHeI_ciHeI;
    const double ciHeII_ciHeII = (data->cell_data[i]).cs_ciHeII_ciHeII;
    const double ciHeIS_ciHeIS = (data->cell_data[i]).cs_ciHeIS_ciHeIS;
    const double ciHI_ciHI = (data->cell_data[i]).cs_ciHI_ciHI;
    const double compton_comp_ = (data->cell_data[i]).cs_compton_comp_;
    const double gloverabel08_gael = (data->cell_data[i]).cs_gloverabel08_gael;
    const double gloverabel08_gaH2 = (data->cell_data[i]).cs_gloverabel08_gaH2;
    const double gloverabel08_gaHe = (data->cell_data[i]).cs_gloverabel08_gaHe;
    const double gloverabel08_gaHI = (data->cell_data[i]).cs_gloverabel08_gaHI;
    const double gloverabel08_gaHp = (data->cell_data[i]).cs_gloverabel08_gaHp;
    const double gloverabel08_h2lte = (data->cell_data[i]).cs_gloverabel08_h2lte;
    const double h2formation_h2mcool = (data->cell_data[i]).cs_h2formation_h2mcool;
    const double h2formation_h2mheat = (data->cell_data[i]).cs_h2formation_h2mheat;
    const double h2formation_ncrd1 = (data->cell_data[i]).cs_h2formation_ncrd1;
    const double h2formation_ncrd2 = (data->cell_data[i]).cs_h2formation_ncrd2;
    const double h2formation_ncrn = (data->cell_data[i]).cs_h2formation_ncrn;
    const double reHeII1_reHeII1 = (data->cell_data[i]).cs_reHeII1_reHeII1;
    const double reHeII2_reHeII2 = (data->cell_data[i]).cs_reHeII2_reHeII2;
    const double reHeIII_reHeIII = (data->cell_data[i]).cs_reHeIII_reHeIII;
    const double reHII_reHII = (data->cell_data[i]).cs_reHII_reHII;
    const double h2_optical_depth_approx = (data->cell_data[i]).h2_optical_depth_approx;
    const double cie_optical_depth_approx = (data->cell_data[i]).cie_optical_depth_approx;

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
  cvklu_data *data = (cvklu_data*)user_data;
  const int NSPARSE = 64;
  const double mh = 1.67e-24;
  const double z = data->current_z;
  double *scale     = data->scale;
  double *inv_scale = data->inv_scale;
  double *ydata = N_VGetArrayPointer(y);

  // Access CSR sparse matrix structures, and zero out data
  sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(J);
  sunindextype *colvals = SUNSparseMatrix_IndexValues(J);
  realtype *matrix_data = SUNSparseMatrix_Data(J);
  SUNMatZero(J);

  // Loop over data, filling in sparse Jacobian
  //  for (int i = 0; i < data->nstrip; i++ ){
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,data->nstrip), [=] RAJA_DEVICE (int i) {

    // Set up some temporaries
    const double T   = (data->cell_data[i]).Ts;
    const double Tge = (data->cell_data[i]).dTs_ge;
    const double k01 = (data->cell_data[i]).rs_k01;
    const double rk01= (data->cell_data[i]).drs_k01;
    const double k02 = (data->cell_data[i]).rs_k02;
    const double rk02= (data->cell_data[i]).drs_k02;
    const double k03 = (data->cell_data[i]).rs_k03;
    const double rk03= (data->cell_data[i]).drs_k03;
    const double k04 = (data->cell_data[i]).rs_k04;
    const double rk04= (data->cell_data[i]).drs_k04;
    const double k05 = (data->cell_data[i]).rs_k05;
    const double rk05= (data->cell_data[i]).drs_k05;
    const double k06 = (data->cell_data[i]).rs_k06;
    const double rk06= (data->cell_data[i]).drs_k06;
    const double k07 = (data->cell_data[i]).rs_k07;
    const double rk07= (data->cell_data[i]).drs_k07;
    const double k08 = (data->cell_data[i]).rs_k08;
    const double rk08= (data->cell_data[i]).drs_k08;
    const double k09 = (data->cell_data[i]).rs_k09;
    const double rk09= (data->cell_data[i]).drs_k09;
    const double k10 = (data->cell_data[i]).rs_k10;
    const double rk10= (data->cell_data[i]).drs_k10;
    const double k11 = (data->cell_data[i]).rs_k11;
    const double rk11= (data->cell_data[i]).drs_k11;
    const double k12 = (data->cell_data[i]).rs_k12;
    const double rk12= (data->cell_data[i]).drs_k12;
    const double k13 = (data->cell_data[i]).rs_k13;
    const double rk13= (data->cell_data[i]).drs_k13;
    const double k14 = (data->cell_data[i]).rs_k14;
    const double rk14= (data->cell_data[i]).drs_k14;
    const double k15 = (data->cell_data[i]).rs_k15;
    const double rk15= (data->cell_data[i]).drs_k15;
    const double k16 = (data->cell_data[i]).rs_k16;
    const double rk16= (data->cell_data[i]).drs_k16;
    const double k17 = (data->cell_data[i]).rs_k17;
    const double rk17= (data->cell_data[i]).drs_k17;
    const double k18 = (data->cell_data[i]).rs_k18;
    const double rk18= (data->cell_data[i]).drs_k18;
    const double k19 = (data->cell_data[i]).rs_k19;
    const double rk19= (data->cell_data[i]).drs_k19;
    const double k21 = (data->cell_data[i]).rs_k21;
    const double rk21= (data->cell_data[i]).drs_k21;
    const double k22 = (data->cell_data[i]).rs_k22;
    const double rk22= (data->cell_data[i]).drs_k22;
    const double brem_brem = (data->cell_data[i]).cs_brem_brem;
    const double rbrem_brem = (data->cell_data[i]).dcs_brem_brem;
    const double ceHeI_ceHeI = (data->cell_data[i]).cs_ceHeI_ceHeI;
    const double rceHeI_ceHeI = (data->cell_data[i]).dcs_ceHeI_ceHeI;
    const double ceHeII_ceHeII = (data->cell_data[i]).cs_ceHeII_ceHeII;
    const double rceHeII_ceHeII = (data->cell_data[i]).dcs_ceHeII_ceHeII;
    const double ceHI_ceHI = (data->cell_data[i]).cs_ceHI_ceHI;
    const double rceHI_ceHI = (data->cell_data[i]).dcs_ceHI_ceHI;
    const double cie_cooling_cieco = (data->cell_data[i]).cs_cie_cooling_cieco;
    const double rcie_cooling_cieco = (data->cell_data[i]).dcs_cie_cooling_cieco;
    const double ciHeI_ciHeI = (data->cell_data[i]).cs_ciHeI_ciHeI;
    const double rciHeI_ciHeI = (data->cell_data[i]).dcs_ciHeI_ciHeI;
    const double ciHeII_ciHeII = (data->cell_data[i]).cs_ciHeII_ciHeII;
    const double rciHeII_ciHeII = (data->cell_data[i]).dcs_ciHeII_ciHeII;
    const double ciHeIS_ciHeIS = (data->cell_data[i]).cs_ciHeIS_ciHeIS;
    const double rciHeIS_ciHeIS = (data->cell_data[i]).dcs_ciHeIS_ciHeIS;
    const double ciHI_ciHI = (data->cell_data[i]).cs_ciHI_ciHI;
    const double rciHI_ciHI = (data->cell_data[i]).dcs_ciHI_ciHI;
    const double compton_comp_ = (data->cell_data[i]).cs_compton_comp_;
    const double rcompton_comp_ = (data->cell_data[i]).dcs_compton_comp_;
    const double gloverabel08_gael = (data->cell_data[i]).cs_gloverabel08_gael;
    const double rgloverabel08_gael = (data->cell_data[i]).dcs_gloverabel08_gael;
    const double gloverabel08_gaH2 = (data->cell_data[i]).cs_gloverabel08_gaH2;
    const double rgloverabel08_gaH2 = (data->cell_data[i]).dcs_gloverabel08_gaH2;
    const double gloverabel08_gaHe = (data->cell_data[i]).cs_gloverabel08_gaHe;
    const double rgloverabel08_gaHe = (data->cell_data[i]).dcs_gloverabel08_gaHe;
    const double gloverabel08_gaHI = (data->cell_data[i]).cs_gloverabel08_gaHI;
    const double rgloverabel08_gaHI = (data->cell_data[i]).dcs_gloverabel08_gaHI;
    const double gloverabel08_gaHp = (data->cell_data[i]).cs_gloverabel08_gaHp;
    const double rgloverabel08_gaHp = (data->cell_data[i]).dcs_gloverabel08_gaHp;
    const double gloverabel08_h2lte = (data->cell_data[i]).cs_gloverabel08_h2lte;
    const double rgloverabel08_h2lte = (data->cell_data[i]).dcs_gloverabel08_h2lte;
    const double h2formation_h2mcool = (data->cell_data[i]).cs_h2formation_h2mcool;
    const double rh2formation_h2mcool = (data->cell_data[i]).dcs_h2formation_h2mcool;
    const double h2formation_h2mheat = (data->cell_data[i]).cs_h2formation_h2mheat;
    const double rh2formation_h2mheat = (data->cell_data[i]).dcs_h2formation_h2mheat;
    const double h2formation_ncrd1 = (data->cell_data[i]).cs_h2formation_ncrd1;
    const double rh2formation_ncrd1 = (data->cell_data[i]).dcs_h2formation_ncrd1;
    const double h2formation_ncrd2 = (data->cell_data[i]).cs_h2formation_ncrd2;
    const double rh2formation_ncrd2 = (data->cell_data[i]).dcs_h2formation_ncrd2;
    const double h2formation_ncrn = (data->cell_data[i]).cs_h2formation_ncrn;
    const double rh2formation_ncrn = (data->cell_data[i]).dcs_h2formation_ncrn;
    const double reHeII1_reHeII1 = (data->cell_data[i]).cs_reHeII1_reHeII1;
    const double rreHeII1_reHeII1 = (data->cell_data[i]).dcs_reHeII1_reHeII1;
    const double reHeII2_reHeII2 = (data->cell_data[i]).cs_reHeII2_reHeII2;
    const double rreHeII2_reHeII2 = (data->cell_data[i]).dcs_reHeII2_reHeII2;
    const double reHeIII_reHeIII = (data->cell_data[i]).cs_reHeIII_reHeIII;
    const double rreHeIII_reHeIII = (data->cell_data[i]).dcs_reHeIII_reHeIII;
    const double reHII_reHII = (data->cell_data[i]).cs_reHII_reHII;
    const double rreHII_reHII = (data->cell_data[i]).dcs_reHII_reHII;

    int j = i * NSPECIES;
    const double H2_1 = ydata[j]*scale[j];
    j++;

    const double H2_2 = ydata[j]*scale[j];
    j++;

    const double H_1 = ydata[j]*scale[j];
    j++;

    const double H_2 = ydata[j]*scale[j];
    j++;

    const double H_m0 = ydata[j]*scale[j];
    j++;

    const double He_1 = ydata[j]*scale[j];
    j++;

    const double He_2 = ydata[j]*scale[j];
    j++;

    const double He_3 = ydata[j]*scale[j];
    j++;

    const double de = ydata[j]*scale[j];
    j++;

    const double ge = ydata[j]*scale[j];
    j++;

    const double mdensity = (data->cell_data[i]).mdensity;
    const double inv_mdensity = 1.0 / mdensity;
    const double h2_optical_depth_approx = (data->cell_data[i]).h2_optical_depth_approx;
    const double cie_optical_depth_approx = (data->cell_data[i]).cie_optical_depth_approx;

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

    //  }
  });

  rowptrs[ data->nstrip * NSPECIES ] = data->nstrip * NSPARSE;
  return 0;
}




void setting_up_extra_variables( cvklu_data * data, double * input, int nstrip ){

  const double mh = 1.67e-24;
  //  for ( int i = 0; i < nstrip; i++){
  RAJA::forall<EXECPOLICY>(RAJA::RangeSegment(0,nstrip), [=] RAJA_DEVICE (int i) {

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

    mdensity *= mh;

    double tau = pow( (mdensity / 3.3e-8 ), 2.8);
    tau = fmax( tau, 1.0e-5 );

    // store results
    (data->cell_data[i]).mdensity = mdensity;
    (data->cell_data[i]).inv_mdensity = 1.0 / mdensity;
    (data->cell_data[i]).cie_optical_depth_approx = fmin( 1.0, (1.0 - exp(-tau) ) / tau );
    (data->cell_data[i]).h2_optical_depth_approx = fmin( 1.0, pow( (mdensity / (1.34e-14) )  , -0.45) );
  //}
  });



}
