#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/srun_setup.sh


#ask for challenge, system, and algorithm
read -p "Challenge "fc","fts","fe"]: " challenge
read -p "System: " system
read -p "Algorithm: " algo

# Find potentially matching params files
parent_dir = $DD_SCRATCH_FOLDER/$DD_OUTPUT_FOLDER/$challenge/$system
# List all subdirectories of parent dir


SUBMIT_SCRIPT="slurm/$DD_CLUSTER/submit.sh"
#check if the submit script exists
if [ ! -f "$SUBMIT_SCRIPT" ]; then
  echo -e "${RED} Submit script $SUBMIT_SCRIPT does not exist ${NC}";
  return 0 2> /dev/null || exit 0
else
    chmod +x $SUBMIT_SCRIPT #make the script executable
    ./$SUBMIT_SCRIPT "$@" #run the submit script passing all arguments
fi