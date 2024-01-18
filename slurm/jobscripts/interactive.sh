#!/bin/bash

# if quest cluster, load singularity module
if [[ $DD_CLUSTER == "quest" ]]; then
    module load singularity
fi

if test -f $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif; then
    echo “docker image exists”
else
    echo "Pulling docker image to $DD_SINGULARITY_IMAGE_LOCATION"
    mkdir -p $DD_SINGULARITY_IMAGE_LOCATION
    singularity pull $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif $DD_IMAGE_REPO
fi

singularity shell \
                --bind $DD_REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $DD_SCRATCH_DIR/$DD_OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $DD_REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $DD_SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif

# Now you can use the Experiments command line interface to make and check experiments
# To run: ./slurm/scripts/srun_interactive.sh