#!/bin/bash

# If running from OnDemand Code Server, might have to run:
#   unset PROMPT_COMMAND

### CHANGE THESE VALUES TO YOUR OWN
SINGULARITY_IMAGE_LOCATION=$GROUP_HOME/$USER/simg    #where you want to store the singularity image
REPO_DIR=$HOME                            #parent directory of where you cloned dynadojo
SCRATCH_DIR=$SCRATCH                         #your scratch directory  
OUTPUT_DIR=sherput                       #name of folder in scratch to put output
IMAGE_REPO=docker://carynbear/dynadojo:sherlock       #docker image to pull
### CHANGE THESE VALUES TO YOUR OWN

if test -f $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif; then
  echo “docker image exists”
else
    mkdir -p $SINGULARITY_IMAGE_LOCATION
    singularity pull $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif $IMAGE_REPO
fi

singularity shell \
                --bind $REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $SCRATCH_DIR/$OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif

# Now you can use the Experiments command line interface to make and check experiments