#!/bin/bash
#SBATCH -n 1 # 1 Cores
#SBATCH -N 1 # All cores on one machine
#SBATCH --mem 4G # Memory (4Gb)
#SBATCH -t 0-2:30 # Maximum execution time (D-HH:MM)
#SBATCH --array=1-100   # maps to SLURM_ARRAY_TASK_ID below

## REQUIRED ENVIRONMENT VARIABLES (see dynadojo/slurm/<cluster>/env.sh for example)
# DD_SINGULARITY_IMAGE_LOCATION                #where you want to store the singularity image
# DD_REPO_DIR                                  #parent directory of where you cloned dynadojo
# DD_SCRATCH_DIR                               #your scratch directory  
# DD_OUTPUT_DIR                                #name of folder in scratch to put output 
# DD_IMAGE_REPO                                #docker image to pull
# DD_CLUSTER                                   #cluster name
#
# Recommended to use dynadojo/slurm/scripts/srun_submit.sh OR sbatch_submit.sh to run this script
#   In order to load the correct environment variables and job params
#   Otherwise, make sure env vars and job params are appropriate and submit:
# 
# sbatch <job-options> make_n_run.sbatch <fc|fts|fe> <system> <model> <optional-jobs-list>
#
# EXAMPLE: sbatch <job-options> make_n_run.sbatch fc lds lr_5 0,1,2,3,4,5,6,7,8,9

CHALLENGE=$1
SYSTEM=$2
ALGO=$3
JOBS=$4 #default to None if not provided

# if quest cluster, load singularity module
if [[ $DD_CLUSTER == "quest" ]]; then
    module load singularity
fi

if [[ $SLURM_ARRAY_TASK_ID == "" ]]; then
    SLURM_ARRAY_TASK_ID=0
fi

if [[ $SLURM_ARRAY_TASK_MAX == "" ]]; then
    SLURM_ARRAY_TASK_MAX=0
fi


singularity run --bind $DD_REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $DD_SCRATCH_DIR/$DD_OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $DD_REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif \
                python -u -m experiments \
                    make \
                        --challenge=$CHALLENGE \
                        --system=$SYSTEM \
                        --algo=$ALGO \
                    run \
                        --node=$SLURM_ARRAY_TASK_ID \
                        --total_nodes=$SLURM_ARRAY_TASK_MAX \
                        --num_cpu_parallel=$SLURM_CPUS_PER_TASK \
                        --jobs=$JOBS


