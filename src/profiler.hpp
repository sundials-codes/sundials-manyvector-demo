/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU
 ----------------------------------------------------------------
 Copyright (c) 2019, Southern Methodist University.
 All rights reserved.
 For details, see the LICENSE file.
 ----------------------------------------------------------------
 Simple MPI-parallel profiler class for manual code
 instrumentation.
 ---------------------------------------------------------------*/

#ifndef __PROFILER_HPP
#define __PROFILER_HPP

// Header files
#include <mpi.h>


// Profiling class
class Profile {

private:

  double   stime;   // process-local start time for this profile

public:

  double   time;    // accumulated process-local time for this profile
  long int count;   // total process-local calls for this profile

  Profile(): stime(-1.0), time(0.0), count(0) {};

  // starts the local profiler clock; returns 0 on success and 1 on failure
  int start()
  {
    if (stime != -1.0)  return 1;
    stime = MPI_Wtime();
    return 0;
  }

  // stops the local profiler clock; returns 0 on success and 1 on failure
  int stop()
  {
    if (stime == -1.0)  return 1;
    double tcur = MPI_Wtime() - stime;
    stime = -1.0;
    count++;
    time += tcur;
    return 0;
  }

  // returns the mean/min/max (over MPI tasks) cumulative profiler times to
  // all calling tasks
  int cumulative_times(MPI_Comm comm, double& tmean, double& tmin, double& tmax)
  {
    int nprocs, retval;
    retval = MPI_Comm_size(comm, &nprocs);
    if (retval != MPI_SUCCESS)  return(1);
    retval = MPI_Allreduce(&time, &tmean, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (retval != MPI_SUCCESS)  return(1);
    retval = MPI_Allreduce(&time, &tmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    if (retval != MPI_SUCCESS)  return(1);
    retval = MPI_Allreduce(&time, &tmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (retval != MPI_SUCCESS)  return(1);
    tmean /= nprocs;
    return(0);
  }

  // returns the mean/min/max (over MPI tasks) average profiler times to
  // all calling tasks
  int average_times(MPI_Comm comm, double& tmean, double& tmin, double& tmax)
  {
    int nprocs, retval;
    double myavg = time / count;
    retval = MPI_Comm_size(comm, &nprocs);
    if (retval != MPI_SUCCESS)  return(1);
    retval = MPI_Allreduce(&myavg, &tmean, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (retval != MPI_SUCCESS)  return(1);
    retval = MPI_Allreduce(&myavg, &tmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    if (retval != MPI_SUCCESS)  return(1);
    retval = MPI_Allreduce(&myavg, &tmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (retval != MPI_SUCCESS)  return(1);
    tmean /= nprocs;
    return(0);
  }

};  // end Profile


#endif

//---- end of file ----
