#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/srun_setup.sh


#ask for challenge, system, and algorithm
read -p "Challenge "fc","fts","fe"]: " challenge
read -p "System: " system
read -p "Algorithm: " algo

# Find potentially matching params files
base_dir = $DD_SCRATCH_FOLDER/$DD_OUTPUT_FOLDER
parent_dir = $base_dir/$challenge/$system
# List all subdirectories of parent dir
subdirs = $(find $parent_dir -maxdepth 1 -type d)
# Find all params.json files in subdirs
params_files = $(find $subdirs -name "params.json")
# Replace base_dir from params_files with "experiments/outputs"
params_files = $(echo $params_files | sed "s|$base_dir|experiments/outputs|g")
# Print out params files as a numbered list for user to select from
echo "Found the following params files:"
for i in "${!params_files[@]}"; do 
    echo "$i: ${params_files[$i]}"
done


# Ask user to select params file & make sure it is a valid selection
echo ""
read -p "Select params file: " params_file_index
while [[ ! $params_file_index =~ ^[0-9]+$ ]] || [[ $params_file_index -ge ${#params_files[@]} ]]; do
    echo "Invalid selection. Try again."
    read -p "Select params file: " params_file_index
done
params_file = ${params_files[$params_file_index]}
echo "Selected params file: $params_file"

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