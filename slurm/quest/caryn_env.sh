#!/bin/bash
# QUEST SLURM SETTINGS
MY_USER='ctb3982'           #edit this
MY_ACCOUNT='p32141'           #edit this, to check which account(s) is(are) associated with your user, e.g. with sacctmgr list user $USER.

eval "ACCOUNT='-A ${MY_ACCOUNT}'"
PARTITION='--partition normal'
eval "OUTPUT='-o /home/${MY_USER}/logs/out/%A_%a.out'"
eval "ERROR='-e /home/${MY_USER}/logs/err/%A_%a.err'"
TIME='-t 0-8:58' # Maximum execution time (D-HH:MM)
MEM='--mem-per-cpu 4G' #Memory limit

# # QUEST SLURM SETTINGS
# ACCOUNT='-A p32141'
# PARTITION='--partition short'
# OUTPUT='-o /home/ctb3982/logs/out/%A_%a.out'
# ERROR='-e /home/ctb3982/logs/err/%A_%a.err'
# TIME='-t 0-2:30' # Maximum execution time (D-HH:MM)

# ENVIRONMENT VARS
export DD_SINGULARITY_IMAGE_LOCATION=$HOME/simg                 #where you want to store the singularity image
export DD_REPO_DIR=$HOME                                        #parent directory of where you cloned dynadojo
export DD_SCRATCH_DIR=$HOME                                     #your scratch directory  
export DD_OUTPUT_DIR=caryn_output                               #name of folder in scratch to put output 
export DD_IMAGE_REPO=docker://carynbear/dynadojo:slurm       #docker image to pull
export DD_CLUSTER=quest                                         #cluster name
export DD_SLURM_ARGS="$ACCOUNT $PARTITION $TIME $MEM"
export DD_SLURM_SAVE="$OUTPUT $ERROR"

echo "Setting up environment for $DD_CLUSTER"