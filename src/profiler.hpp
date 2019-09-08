/*---------------------------------------------------------------
 Programmer(s): Daniel R. Reynolds @ SMU, Cody J. Balos @ LLNL
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
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

// Profiling class
class Profile {

private:

  std::string description; // profile description
  double   stime;          // process-local start time for this profile

public:

  double   time;    // accumulated process-local time for this profile
  long int count;   // total process-local calls for this profile

  static const int PRINT_ALL_RANKS = -1;

  Profile(): stime(-1.0), time(0.0), count(0), description("") {};

  void desc(const std::string& str)
  {
    description = str;
  }

  std::string desc() const
  {
    return description;
  }

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

  void print_average_times(const std::string& str = "", MPI_Comm comm = MPI_COMM_WORLD,
                           std::ostream& outf = std::cout, int outrank = 0)
  {
    int rank;
    double tmean, tmin, tmax;

    MPI_Comm_rank(comm, &rank);
    bool output = (outrank == PRINT_ALL_RANKS) ? true : (rank == outrank);

    int err = average_times(comm, tmean, tmin, tmax);
    if (err && output) {
      std::cerr << "ERROR: Profile::print_average_times()" << std::endl;
    } else if (output) {
      outf << "Average " << str << " time = \t" << std::scientific << std::setprecision(2)
           << tmean << "  ( min / max  =  " << tmin << " / " << tmax << " )" << std::endl;
    }
  }

  void print_average_times(MPI_Comm comm = MPI_COMM_WORLD, std::ostream& outf = std::cout, int outrank = 0)
  {
    int rank;
    double tmean, tmin, tmax;

    MPI_Comm_rank(comm, &rank);
    bool output = (outrank == PRINT_ALL_RANKS) ? true : (rank == outrank);

    int err = average_times(comm, tmean, tmin, tmax);
    if (err && output) {
      std::cerr << "ERROR: Profile::print_average_times()" << std::endl;
    } else if (output) {
      outf << "Average " << description << " time = \t" << std::scientific << std::setprecision(2)
           << tmean << "  ( min / max  =  " << tmin << " / " << tmax << " )" << std::endl;
    }
  }

  void print_cumulative_times(const std::string& str = "", MPI_Comm comm = MPI_COMM_WORLD,
                              std::ostream& outf = std::cout, int outrank = 0)
  {
    int rank;
    double tmean, tmin, tmax;

    MPI_Comm_rank(comm, &rank);
    bool output = (outrank == PRINT_ALL_RANKS) ? true : (rank == outrank);

    int err = cumulative_times(comm, tmean, tmin, tmax);
    if (err && output) {
      std::cerr << "ERROR: Profile::print_cumulative_times()" << std::endl;
    } else if (output) {
      outf << "Total " << str << " time = \t" << std::scientific << std::setprecision(2)
           << tmean << "  ( min / max  =  " << tmin << " / " << tmax << " )" << std::endl;
    }
  }

  void print_cumulative_times(MPI_Comm comm = MPI_COMM_WORLD, std::ostream& outf = std::cout, int outrank = 0)
  {
    int rank;
    double tmean, tmin, tmax;

    MPI_Comm_rank(comm, &rank);
    bool output = (outrank == PRINT_ALL_RANKS) ? true : (rank == outrank);

    int err = cumulative_times(comm, tmean, tmin, tmax);
    if (err && output) {
      std::cerr << "ERROR: Profile::print_cumulative_times()" << std::endl;
    } else if (output) {
      outf << "Total " << description << " time = \t" << std::scientific << std::setprecision(2)
           << tmean << "  ( min / max  =  " << tmin << " / " << tmax << " )" << std::endl;
    }
  }

  void export_to_file(std::ofstream& outf, MPI_Comm comm = MPI_COMM_WORLD, int outrank = 0)
  {
    int rank;
    double tmean, tmin, tmax;

    MPI_Comm_rank(comm, &rank);
    bool output = (outrank == PRINT_ALL_RANKS) ? true : (rank == outrank);

    outf << description << std::endl;
    outf << "averaged mean," << "averaged min,"  << "averaged max,"
         << "cumulative mean," << "cumulative min," << "cumulative max" << std::endl;

    int err = average_times(comm, tmean, tmin, tmax);
    if (err && output) {
      std::cerr << "ERROR: Profile::export_to_file()" << std::endl;
    } else if (output) {
      outf << tmean << "," << tmin  << "," << tmax << ",";
    }
    err = cumulative_times(comm, tmean, tmin, tmax);
    if (err && output) {
      std::cerr << "ERROR: Profile::export_to_file()" << std::endl;
    } else if (output) {
      outf << tmean << "," << tmin  << "," << tmax << std::endl;
    }
  }

};  // end Profile


#endif

//---- end of file ----
