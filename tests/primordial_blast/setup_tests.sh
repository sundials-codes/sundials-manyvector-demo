#!/bin/bash
# ------------------------------------------------------------------------------
# Programmer(s): Daniel R. Reynolds @ SMU
#                David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# Copyright (c) 2019
# Southern Methodist University and Lawrence Livermore National Security.
# All rights reserved.
# For details, see the LICENSE file.
# ------------------------------------------------------------------------------
# This script sets up input files for explicit method weak scaling tests for the
# primordial_blast_mr.exe or primordial_blast_imex.exe test codes. Here, the
# mesh is refined in proportion to the number of parallel tasks. However, what
# makes it "explicit" is that the slow time step size is decreased in proportion
# with the spatial mesh size (to satisfy stability considerations), and the
# simulation is run for a fixed number of "slow" time steps (instead of to a
# fixed final time).
#
# In these tests, the fast time scale is evolved with a time step that is
# independent of the mesh, meaning that as the domain is refined, the multirate
# time scale separation factor decreases. This can be modified by changing the
# "hmax=..." line below to use the currently-commented version.
#
# Two inputs are required:
#   1. The location where the testing files should be written.
#   2. The test type: multirate or imex
#
# One optional input is supported:
#   1. The h_fast type: fixed or refined (default refined)
#
# Example usage:
#
# summit: ./setup_tests.sh $PROJWORK/project/ManyVector-demo-runs multirate
# lassen: ./setup_tests.sh /p/gpfs1/user/ManyVector-demo-runs imex
# ------------------------------------------------------------------------------

# basic variables
TasksPerNode=40    # number of MPI tasks to use per node
base_nodes=2       # smallest number of nodes to test
base_npx=5         # base 80-task Cartesian decomposition
base_npy=4
base_npz=4
base_nx=25         # base grid on each process
base_ny=25
base_nz=25
base_tf=0.1        # base final simulation time
base_h0=0.01       # base slow time step size
base_m=100.0       # base fast/slow time scale separation factor
base_htrans=0.01   # base fast transient evolution interval

# testing 'dimensions' Summit
#NodeFactor=(1 2 4 6 8 10 12)
#Fused=(fused unfused)

# testing 'dimensions' Lassen
NodeFactor=(1 2 3 4 5)
Fused=(fused)

# ------------------------------------------------------------------------------
# Generate test files
# ------------------------------------------------------------------------------

# check for required inputs
if [ "$#" -lt 2 ]; then
    echo "ERROR: Two inputs required:"
    echo "  1. Location where test files should be written"
    echo "  2. Test type: multirate or imex"
    exit 1
fi
testroot=$1
testtype=$2

# check for valid test type
if [ "$testtype" == "multirate" ]; then
    suffix=mr
elif [ "$testtype" == "imex" ]; then
    suffix=imex
else
    echo "ERROR: Invalid test type: $testtype"
    exit 1
fi

# set defaults for optional inputs
hfasttype=refined

# check for optional inputs
if [ "$#" -gt 2 ]; then
    hfasttype=$3
fi

# check for valid hfast type
case "$hfasttype" in
    fixed|refined) ;;
    *)
        echo "ERROR: Unknown h_fast type: $hfasttype"
        exit 1
        ;;
esac

# check that the HOST environment variable is valid
case "$HOST" in
    summit|lassen) ;;
    *)
        echo "ERROR: Unknown host name: $HOST"
        exit 1
        ;;
esac

# set up each test
for nf in "${NodeFactor[@]}"; do

    for f in "${Fused[@]}"; do

        # set total number of nodes
        let "n = $base_nodes * $nf * $nf * $nf"

        # set directory name for test
        d=n${n}_${f}

        # path the test directory
        testdir=$testroot/$d

        # compute Cartesian parallel decomposition
        let "npx = $base_npx * $nf"
        let "npy = $base_npy * $nf"
        let "npz = $base_npz * $nf"

        # compute global computational grid dimensions
        let "nx = $base_nx * $npx"
        let "ny = $base_ny * $npy"
        let "nz = $base_nz * $npz"

        # compute time-stepping parameters
        tfinal=`python -c "print $base_tf / $nf"`
        h_slow=`python -c "print $base_h0 / $nf"`
        if [ "$hfasttype" == "fixed" ]; then
            h_fast=`python -c "print $base_h0 / $base_m"`  # fixed hfast
        else
            h_fast=`python -c "print $base_h0 / $base_m / $nf"`  # refined hfast
        fi
        h_transient=`python -c "print $base_htrans / $nf"`

        # compute total number of MPI tasks
        let "m = $TasksPerNode * $n"

        # sanity check for agreement
        let "m2 = $npx * $npy * $npz"
        if [ $m -ne $m2 ]; then
            echo "setup_tests.sh error, $m does not equal $m2"
            exit 1
        fi

        # create directory skeleton
        echo "Setting up test $d: $n nodes, $m tasks ($f vectors):"
        echo "    grid = $nx x $ny x $nz"
        echo "    final time = $tfinal"
        echo "    h_slow = $h_slow"
        echo "    h_fast = $h_fast"
        echo "    h_transient = $h_transient"
        echo "  "
        if [ -d $testdir ]; then
            rm -rf $testdir
        fi
        mkdir -p $testdir
        cp primordial_tables.h5 $testdir/
        cp jobscript_${HOST}.lsf $testdir/
        cp input_primordial_blast_${suffix}.txt $testdir/
        cp ../../primordial_blast_${suffix}.exe $testdir/

        # modify input file based on options:
        #   nx, ny, nz -- global problem grid size
        #   tf -- final simulation time
        #   h0 -- slow time step size
        #   hmax -- fast time step size
        #   htrans -- initial interval for fast transient evolution
        #   fusedkernels, localreduce -- flags to enable new N_Vector operations
        inputs=$testdir/input_primordial_blast_${suffix}.txt
        sed -i "/nx =.*/ s/.*#/nx = $nx #/" $inputs
        sed -i "s/ny =.*/ny = $ny/" $inputs
        sed -i "s/nz =.*/nz = $nz/" $inputs
        sed -i "s/tf =.*/tf = $tfinal/" $inputs
        if [ "$testtype" == "multirate" ]; then
            sed -i "/h0 =.*/ s/.*#/h0 = $h_slow #/" $inputs
        fi
        sed -i "/hmax =.*/ s/.*#/hmax = $h_fast #/" $inputs
        sed -i "/htrans =.*/ s/.*#/htrans = $h_transient #/" $inputs
        if [ $f == fused ]; then
            sed -i "/fusedkernels =.*/ s/.*#/fusedkernels = 1 #/" $inputs
            sed -i "/localreduce =.*/ s/.*#/localreduce = 1 #/" $inputs
        else
            sed -i "/fusedkernels =.*/ s/.*#/fusedkernels = 0 #/" $inputs
            sed -i "/localreduce =.*/ s/.*#/localreduce = 0 #/" $inputs
        fi
        # disable output
        sed -i "/nout =.*/ s/.* #/nout = 0 #/" $inputs
        sed -i "/showstats =.*/ s/.*#/showstats = 0 #/" $inputs

        # modify submission script for this job:
        #   test suffix: SUFFIX -> {mr, imex}
        #   max wall clock time (hours): WTIME -> {2,4,6,8}
        #   requested nodes: NODES -> ${n}
        #   job name: NAME -> pbmr-${d}
        #   run directory: RUNDIR -> ${d}
        #   number of resource sets: RESOURCESETS -> ${n} x 2
        jobscript=$testdir/jobscript_${HOST}.lsf
        sed -i "s/SUFFIX/${suffix}/g" $jobscript
        sed -i "s/NODES/${n}/g" $jobscript
        sed -i "s/NAME/pb-${suffix}-${d}/g" $jobscript
        sed -i "s|RUNDIR|${testdir}|g" $jobscript
        let "rs = $n * 2"
        sed -i "s/RESOURCESETS/${rs}/g" $jobscript
        if [ $nf -lt 3 ]; then
            sed -i "s/WTIME/2/g" $jobscript
        elif [ $nf -lt 4 ]; then
            sed -i "s/WTIME/4/g" $jobscript
        elif  [ $nf -lt 8 ]; then
            sed -i "s/WTIME/6/g" $jobscript
        else
            sed -i "s/WTIME/8/g" $jobscript
        fi

    done

done
