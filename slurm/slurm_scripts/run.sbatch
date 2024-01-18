#!/bin/bash
#SBATCH -J dynadojo_run # A single job name for the array
#SBATCH -n 1 # Four Cores
#SBATCH -N 1 # All cores on one machine
#SBATCH --mem 4G # Memory (4Gb)
#SBATCH -t 0-2:30 # Maximum execution time (D-HH:MM)
#SBATCH --array=1-500%100   # maps to SLURM_ARRAY_TASK_ID below

PARAMS_FILE=$1
JOBS="${2:-None}" #default to None if not provided

# if quest cluster, load singularity module
if [[ $DD_CLUSTER == "quest" ]]; then
    module load singularity
fi


singularity run --bind $DD_REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $DD_SCRATCH_DIR/$DD_OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $DD_REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif \
                python -u -m experiments \
                    run \
                        --params_file=$PARAMS_FILE \
                        --output_dir=/$DD_OUTPUT_DIR \
                        --node=$SLURM_ARRAY_TASK_ID \
                        --total_nodes=$SLURM_ARRAY_TASK_MAX \
                        --num_cpu_parallel=$SLURM_CPUS_PER_TASK \
                        --jobs=$JOBS

# Use submit.sh to run this script!!
# To run: ./slurm/sherlock/submit run.sbatch /questput/fc/lds/fc_lds_lr_5_l=5/params.json
# To run: ./slurm/quest/submit    --array=1-{NUMJOBS}%100 run.sbatch /questput/fc/lds/fc_lds_lr_5_l=5/params.json "0,1,2,3,4,5,6,7,8,9" 