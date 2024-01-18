#!/bin/bash

CHALLENGE=$1
SYSTEM=$2
ALGO=$3

# if quest cluster, load singularity module
if [[ $DD_CLUSTER == "quest" ]]; then
    module load singularity
fi

if test -f $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif; then
  echo “docker image exists”
else
    mkdir -p $DD_SINGULARITY_IMAGE_LOCATION
    singularity pull $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif $DD_IMAGE_REPO
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
                        --algo=$ALGO

# To run: use slurm/scripts/srun_make.sh