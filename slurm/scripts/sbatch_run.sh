#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/srun_setup.sh


#ask for challenge, system, and algorithm
read -p "Challenge "fc","fts","fe"]: " challenge
read -p "System: " system
read -p "Algorithm: " algo

# Find potentially matching params files
base_dir=$DD_SCRATCH_DIR/$DD_OUTPUT_DIR
parent_dir=$base_dir/$challenge/$system
# List all subdirectories of parent dir
subdirs=$(find $parent_dir -maxdepth 1 -type d)
# Filter only folders that match the algo $challenge_$system_$algo*
subdirs=$(echo $subdirs | grep $challenge_$system_$algo)
# Find all params.json files in subdirs
params_files=$(find $subdirs -name "params.json")
# Replace base_dir from params_files with "experiments/outputs"
options=( $(echo $params_files | sed "s|$base_dir|experiments/outputs|g") )

# Exit if no options found
if [ ${#options[@]} -eq 0 ]; then
    echo "No params files found in $parent_dir"
    return 0 2> /dev/null || exit 0
fi

# Print out params files as a numbered list for user to select from
echo "Found the following params files:"
select opt in "${options[@]}"
do
    echo "Selected params file: $opt"
    break
done

params_file=$opt

# Ask if they want to supply a list of jobs to run
echo ""
read -p "List of jobs to run (comma separated, no spaces): " jobs

SUBMIT_SCRIPT="slurm/$DD_CLUSTER/submit.sh"
#check if the submit script exists
if [ ! -f "$SUBMIT_SCRIPT" ]; then
  echo -e "${RED} Submit script $SUBMIT_SCRIPT does not exist ${NC}";
  return 0 2> /dev/null || exit 0
else
    chmod +x $SUBMIT_SCRIPT #make the script executable
    ./$SUBMIT_SCRIPT ${__dir}/../jobscripts/sbatch/run.sbatch $params_file $jobs
fi