
#!/bin/bash

#Run the setup script to check the directory and set the environment variables
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${__dir}/_setup_caryn.sh

# Warn user that the directory must be relative to the singularity container
echo "Warning: The data directory must be relative to the singularity container."

#ask for challenge, system, and algorithm
read -p "Challenge "fc","fts","fe"]: " challenge
read -p "System: " system
read -p "Algorithm: " algo

# Find potentially matching data dirs
base_dir=$DD_SCRATCH_DIR/$DD_OUTPUT_DIR
parent_dir=$base_dir/$challenge/$system
# List all subdirectories of parent dir that match the algo $challenge_$system_$algo*
subdirs=$(find $parent_dir -maxdepth 1 -type d | grep $challenge_$system_$algo)
# Replace base_dir from params_files with "experiments/outputs"
options=( $(echo $subdirs | sed "s|$base_dir|experiments/outputs|g") )

# Exit if no options found
if [ ${#options[@]} -eq 0 ]; then
    echo "No params files found in $parent_dir"
    return 0 2> /dev/null || exit 0
fi

# Print out data directories as a numbered list for user to select from
echo "Found the following results directories:"
select opt in "${options[@]}"
do
    echo "Selected results directory: $opt"
    break
done

data_dir=$opt

read -p "Which command? [plot,check]: " cmd
while [[ $cmd != "plot" && $cmd != "check" ]]; do
    read -p "Which command? [plot,check]: " cmd
done


chmod +x slurm/jobscripts/$cmd.sh #make the script executable
srun $DD_SLURM_ARGS --export=all -c 1 slurm/jobscripts/$cmd.sh $data_dir