/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Test problem description
---------------------------------------------------------------*/

// Header files
#include <euler3D.hpp>


// Initial conditions
int initial_conditions(const realtype& t, N_Vector w, const EulerData& udata)
{
  return 0;
}

// External forcing terms
int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  return 0;
}

// Diagnostics output for this test
int output_diagnostics(const realtype& t, const N_Vector w, const EulerData& udata)
{
  return(0);
}

//---- end of file ----
