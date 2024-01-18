### CHANGE THESE VALUES TO YOUR OWN
SINGULARITY_IMAGE_LOCATION=$HOME/simg     #where you want to store the singularity image
REPO_DIR=$HOME                            #parent directory of where you cloned dynadojo
SCRATCH_DIR=$HOME                         #your scratch directory  
OUTPUT_DIR=questput                       #name of folder in scratch to put output
### CHANGE THESE VALUES TO YOUR OWN

CHALLENGE=$1
SYSTEM=$2
ALGO=$3
JOBS="${4:-None}" #default to None if not provided

singularity run --bind $REPO_DIR/dynadojo/experiments:/dynadojo/experiments \
                --bind $SCRATCH_DIR/$OUTPUT_DIR:/dynadojo/experiments/outputs \
                --bind $REPO_DIR/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo \
                --pwd /dynadojo \
                $SINGULARITY_IMAGE_LOCATION/dynadojo_sherlock.sif \
                python -u -m experiments \
                    make \
                        --challenge=$CHALLENGE \
                        --system=$SYSTEM \
                        --algo=$ALGO \
                    run \
                        --output_dir=/$OUTPUT_DIR \
                        --node=$SLURM_ARRAY_TASK_ID \
                        --total_nodes=$SLURM_ARRAY_TASK_MAX \
                        --num_cpu_parallel=$SLURM_CPUS_PER_TASK \
                        --jobs=$JOBS