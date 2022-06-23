#!/bin/bash
# ------------------------------------------------------------------------------
# Programmer(s): Daniel R. Reynolds @ SMU
#                David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# Copyright (c) 2022
# Southern Methodist University and Lawrence Livermore National Security.
# All rights reserved.
# For details, see the LICENSE file.
# ------------------------------------------------------------------------------
# This script sets up run folders for weak scaling tests for allocation of the 
# magma dense block matrix and linear solver.
#
# One input is required: the location where the testing files should be written.
#
# Example usage:
#
# summit: ./setup_tests.sh $PROJWORK/project/magma-scaling-runs
# lassen: ./setup_tests.sh /p/gpfs1/user/magma-scaling-runs
# ------------------------------------------------------------------------------

# basic variables
TasksPerNode=6     # number of MPI tasks to use per node
base_nodes=4       # smallest number of nodes to test

# testing 'dimensions'
NodeFactor=(1 2 4 6 8 10)

# ------------------------------------------------------------------------------
# Generate test files
# ------------------------------------------------------------------------------

# check for required inputs
if [ "$#" -lt 1 ]; then
    echo "ERROR: One input required: location where test files should be written"
    exit 1
fi
testroot=$1

# set HOST if unset on OLCF systems
HOST=${HOST:-$LMOD_SYSTEM_NAME}

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

    # set total number of nodes
    let "n = $base_nodes * $nf * $nf * $nf"

    # set directory name for test
    d=n${n}

    # path the test directory
    testdir=$testroot/$d

    # compute total number of MPI tasks
    let "m = $TasksPerNode * $n"

    # create directory skeleton
    echo "Setting up test $d: $n nodes, $m tasks"
    if [ -d $testdir ]; then
        rm -rf $testdir
    fi
    mkdir -p $testdir
    cp jobscript_${HOST}.lsf $testdir/
    cp ../../magma_scaling.exe $testdir/

    # modify submission script for this job:
    #   max wall clock time (hours): WTIME -> {2,4,6,8}
    #   requested nodes: NODES -> ${n}
    #   job name: NAME -> magma-scaling-${d}
    #   run directory: RUNDIR -> ${d}
    #   number of resource sets: RESOURCESETS -> ${n} x 2
    jobscript=$testdir/jobscript_${HOST}.lsf
    sed -i "s/NODES/${n}/g" $jobscript
    sed -i "s/NAME/magma-scaling-${d}/g" $jobscript
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
