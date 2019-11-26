#!/bin/bash
#
# This script sets up input files for explicit method weak scaling
# tests for the primordial_blast_mr.exe test code.  Here, the mesh
# size is increased in proportion to the number of parallel tasks.
# However, what makes it "explicit" is that the slow time step size
# is decreased in proportion with the spatial mesh size (to satisfy
# stability considerations), and the simulation is run for a fixed
# number of "slow" time steps (instead of to a fixed final time).
#
# In these tests, the fast time scale is evolved with a time step
# that is independent of the mesh, meaning that as the domain is
# refined, the multirate time scale separation factor decreases.
# This can be modified by changing the "hmax=..." line below to
# use the currently-commented version.
#
# Daniel R. Reynolds @ SMU
# September 18, 2019

# basic variables
TasksPerNode=40    # number of MPI tasks to use per node
base_nodes=2       # smallest number of nodes to test
base_npx=5         # base 80-task Cartesian decomposition
base_npy=4
base_npz=4
base_nx=25         # base grid on each process
base_ny=25
base_nz=25
base_tf=1.0        # base final simulation time
base_h0=0.1        # base slow time step size
base_m=1000.0      # base fast/slow time scale separation factor
base_htrans=0.1    # base fast transient evolution interval

# testing 'dimensions'
NodeFactor=(1 2 4 6 8 10 12)
Fused=(fused unfused)

# ------------------------------------------------------------------------------
# Generate test files
# ------------------------------------------------------------------------------

# check for output directory
if [ "$#" -lt 1 ]; then
    echo "ERROR: Path to testing directory required"
    exit 1
fi
testroot=$1

# check that the HOST environment variable is valid
case "$HOST" in
    summit|lassen) ;;
    *)
        echo "ERROR: Unknown host: $HOST"
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
        #h_fast=`python -c "print $base_h0 / $base_m"`  # fixed hfast
        h_fast=`python -c "print $base_h0 / $base_m / $nf"`  # refined hfast
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
        cp input_primordial_blast_mr.txt $testdir/
        cp primordial_tables.h5 $testdir/
        cp jobscript_${HOST}.lsf $testdir/
        cp primordial_blast_mr.exe $testdir/

        # add specific flags to input file based on options:
        #   nx, ny, nz -- global problem grid size
        #   tf -- final simulation time
        #   h0 -- slow time step size
        #   hmax -- fast time step size
        #   htrans -- initial interval for fast transient evolution
        #   fusedkernels, localreduce -- flags to enable new N_Vector operations
        inputs=$testdir/input_primordial_blast_mr.txt
        sed -i "/nx =.*/ s/.*#/nx = $nx #/" $inputs
        sed -i "s/ny =.*/ny = $ny/" $inputs
        sed -i "s/nz =.*/nz = $nz/" $inputs
        sed -i "s/tf =.*/tf = $tfinal/" $inputs
        sed -i "/h0 =.*/ s/.*#/h0 = $h_slow #/" $inputs
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

        # modify submission script for this job
        #   max wall clock time (hours): WTIME -> {2,4,6,8}
        #   requested nodes: NODES -> ${n}
        #   job name: NAME -> pbmr-${d}
        #   run directory: RUNDIR -> ${d}
        #   number of resource sets: RESOURCESETS -> ${n} x 2
        jobscript=$testdir/jobscript_${HOST}.lsf
        sed -i "s/NODES/${n}/g" $jobscript
        sed -i "s/NAME/pbmr-${d}/g" $jobscript
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
