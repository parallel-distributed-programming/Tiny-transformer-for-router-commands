#!/bin/bash
#PBS -q lecture-mig
#PBS -l select=1
#PBS -l walltime=12:00
#PBS -W group_list=gt81
#PBS -j oe
# Build and run the serial baseline on a compute node (submit from this directory):
#   qsub run_job.sh
module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
cd "$PBS_O_WORKDIR"
make -s && ./check && /usr/bin/time -f "wall_s=%e" ./tinygpt
