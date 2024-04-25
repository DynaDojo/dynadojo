#!/bin/bash

# Make a copy of this file in this folder (`dynadojo/slurm/quest`) and name it as `env.sh`.

# QUEST SLURM SETTINGS
MY_USER='<YOUR STUDENT ID>'           #edit this
MY_ACCOUNT='<YOUR ACCOUNT>'           #edit this, to check which account(s) is(are) associated with your user, e.g. with sacctmgr list user $USER.

eval "ACCOUNT='-A ${MY_ACCOUNT}'"
PARTITION='--partition short'
eval "OUTPUT='-o /home/${MY_USER}/logs/out/%A_%a.out'"
eval "ERROR='-e /home/${MY_USER}/logs/err/%A_%a.err'"
TIME='-t 0-2:30' # Maximum execution time (D-HH:MM)

# ENVIRONMENT VARS
export DD_SINGULARITY_IMAGE_LOCATION=$HOME/simg                 #where you want to store the singularity image
export DD_REPO_DIR=$HOME                                        #parent directory of where you cloned dynadojo
export DD_SCRATCH_DIR=$HOME                                     #your scratch directory  
export DD_OUTPUT_DIR=questput                                   #name of folder in scratch to put output 
export DD_IMAGE_REPO=docker://carynbear/dynadojo:slurm          #docker image to pull
export DD_CLUSTER=quest                                         #cluster name
export DD_SLURM_ARGS="$ACCOUNT $PARTITION $TIME"
export DD_SLURM_SAVE="$OUTPUT $ERROR"

echo "Setting up environment for $DD_CLUSTER"