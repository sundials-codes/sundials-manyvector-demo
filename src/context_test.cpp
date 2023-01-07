/* -----------------------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 * -----------------------------------------------------------------------------
 * Simple error reporducer
 * ---------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <mpi.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_nvector.h>

using namespace std;

class EulerData { };

int initial_conditions(const realtype& t, N_Vector w, const EulerData& udata)
{
  return 0;
}

int external_forces(const realtype& t, N_Vector G, const EulerData& udata)
{
  return 0;
}

int output_diagnostics(const realtype& t, const N_Vector w, const EulerData& udata)
{
  return 0;
}

int main(int argc, char* argv[])
{
  // initialize MPI
  int myid, retval;
  retval = MPI_Init(&argc, &argv);
  if (retval) return 1;
  retval = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (retval) MPI_Abort(MPI_COMM_WORLD, 1);

  SUNContext ctx = nullptr;
  MPI_Comm comm = MPI_COMM_WORLD;

  if (myid == 0)
  {
    cout << "MPI_COMM_WORLD = " << MPI_COMM_WORLD << endl;
    cout << "comm           = " << comm << endl;
    cout << "ctx            = " << ctx << endl;
  }

  if (SUNContext_Create((void*) &comm, &ctx))
  {
    cout << "ERROR: SUNContext_Create 1 Failed" << endl;
  }
  else
  {
    cout << "SUCCESS: SUNContext_Create 1 Passed" << endl;
    SUNContext_Free(&ctx);
  }

  if (SUNContext_Create(nullptr, &ctx))
  {
    cout << "ERROR: SUNContext_Create 2 Failed" << endl;
  }
  else
  {
    cout << "SUCCESS: SUNContext_Create 2 Passed" << endl;
    SUNContext_Free(&ctx);
  }

  MPI_Finalize();

  return 0;
}

//---- end of file ----
