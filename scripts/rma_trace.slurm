#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP_PROJ01_DEBUG2
#SBATCH -p qcpu
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --distribution=block:cyclic:cyclic,NoPack

source load_modules.sh

STDOUT_FILE="run_debug_1d_out.csv"
STDERR_FILE="run_debug_1d_err.txt"
BINARY_PATH="../build_prof/ppp_proj01"

size=2048
procs=4

export KMP_AFFINITY=compact

# Clear the stdout and stderr files
rm -f $STDOUT_FILE $STDERR_FILE

DISK_WRITE_INTENSITY=100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# USER_SCRATCH_PATH=/scratch/project/dd-23-135/$USER

# mkdir -p $USER_SCRATCH_PATH

# OUT_FILE_PATH=$USER_SCRATCH_PATH/$SLURM_JOBID

# mkdir -p $OUT_FILE_PATH


# Doplnte vhodne nastavenie Lustre file system #
################################################

# lfs setstripe -S 2M -c 16 /scratch/project/dd-23-135/$USER

################################################


nnodes=$((procs/SLURM_NTASKS_PER_NODE))

export SCOREP_ENABLE_PROFILING=true
export SCOREP_TOTAL_MEMORY=2G
# srun -n $procs -N $nnodes $BINARY_PATH -B    -n 100 -t $OMP_NUM_THREADS -m 2 -w $DISK_WRITE_INTENSITY -i "input_data_$size.h5" >> $STDOUT_FILE 2>> $STDERR_FILE

export SCOREP_ENABLE_TRACING=true
export SCOREP_FILTERING_FILE=ppp_scorep_filter.flt
srun -n $procs -N $nnodes $BINARY_PATH -B    -n 100 -t $OMP_NUM_THREADS -m 2 -w $DISK_WRITE_INTENSITY -i "input_data_$size.h5" >> $STDOUT_FILE 2>> $STDERR_FILE


# rm -rf $OUT_FILE_PATH
