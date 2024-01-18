#!/bin/bash

DATA_DIR=$1

#if quest cluster, load singularity module
if [[ $DD_CLUSTER == "quest" ]]; then
    module load singularity
fi

singularity run --bind $DD_REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $DD_SCRATCH_DIR/$DD_OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $DD_REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif \
                python -u -m experiments \
                    check \
                        --data_dir=$DATA_DIR

# To run: ./slurm/scripts/srun_check.sh