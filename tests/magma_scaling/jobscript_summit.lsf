#!/bin/bash
#
#BSUB -P csc317                  # charge csc317 project
#BSUB -W WTIME:00                # max wall time (hours)
#BSUB -nnodes NODES              # requested nodes
#BSUB -J NAME                    # job name
#BSUB -o stdout-%J.txt
#BSUB -e stderr-%J.txt
#BSUB -N                         # send email on job completion

# change to run directory
cd RUNDIR

# write timestamp to stdout
echo "  "
date

# ensure that appropriate modules are loaded
module load cuda/11.4.2 gcc/10.2.0 cmake hdf5

# write loaded modules (sent to stderr)
module list

# set the 'jsrun' options [default value]:
#    number of resource sets  [all available physical cores]
JSRUN_N=RESOURCESETS
#    number of MPI tasks per resource set  [no default]
JSRUN_A=3
#    number of CPUs per resource set  [1]
JSRUN_C=3
#    number of GPUs per resource set  [0]
JSRUN_G=3
#    binding of tasks within a resource set (none / rs / packed:#)  [packed:1]
JSRUN_B=packed:1
#    number of resource sets per host  [no default]
JSRUN_R=2
#    latency priority (CPU-CPU / GPU-CPU)  [GPU-CPU,CPU-MEM,CPU-CPU]
JSRUN_L=GPU-CPU
#    how tasks are started on resource sets  [packed]
JSRUN_D=packed

# create jsrun command
CMD="jsrun -n $JSRUN_N -a $JSRUN_A -c $JSRUN_C -g $JSRUN_G -b $JSRUN_B -r $JSRUN_R -l $JSRUN_L -d $JSRUN_D ./magma_scaling.exe"

# echo jsrun command to stdout
echo "  "
echo "jsrun command line:"
echo $CMD

# run the job
echo "  "
$CMD

# copy output/error files to project home folder and update permissions
cp stdout*.txt stderr*.txt /ccs/proj/csc317/magma-scaling-results/
chgrp csc317 /ccs/proj/csc317/magma-scaling-results/*.txt
chmod g+rw /ccs/proj/csc317/magma-scaling-results/*.txt
