#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/srun_setup.sh

SUBMIT_SCRIPT="slurm/$DD_CLUSTER/submit.sh"
#check if the submit script exists
if [ ! -f "$SUBMIT_SCRIPT" ]; then
  echo -e "${RED} Submit script $SUBMIT_SCRIPT does not exist ${NC}";
  return 0 2> /dev/null || exit 0
else
    chmod +x $SUBMIT_SCRIPT #make the script executable
    ./$SUBMIT_SCRIPT "$@" #run the submit script passing all arguments
fi