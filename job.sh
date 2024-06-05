#!/bin/bash

#SBATCH --job-name=matrix-mult
#SBATCH --output=slurmout.txt
#SBATCH --error=slurmerr.txt
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR


srun singularity run --bind $TMPDIR:/scratch_local matrix_multi.sif
