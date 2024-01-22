At the moment, our slurm utilities are written specifically for Stanford's Sherlock and Northwestern's Quest clusters. We hope to eventually write code that can more easily be used by other clusters. For now, please use this code as a reference for porting the code to work for your system. 

# Tips
- Always run from dynadojo folder!
- If you want to skip the prompt for which cluster, set the env var $DD_CLUSTER to "sherlock" or "quest" (in bashrc or however)
- If you encounter any `Permission denied` messages, make the file executable. `chmod +x <file>`
- [Profile](https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1964#section-required-memory) your jobs to test how long/how much memory it takes to run a single job.
- To monitor status, you can also use the watch command. I watch the output directory `watch -n 60 ls -v -C` for a little bit just to check everything is running. or (`watch -n 20 "ls -v . | tail -10"`)

# Setup
1. You *must* clone dynadojo from github into a location that is shared to all nodes, usually $HOME. The singularity container does NOT contain a copy of the code, only the dependencies. The git repo gets mounted onto the container by specifying the $DD_REPO_DIR env var (see `dynadojo/slurm/<cluster>/env.sh`). We do this so you can live-edit your params in `dynadojo/experiments/params.py`.
2. Edit the submission flags and environment variables in `dynadojo/slurm/<cluster>/env.sh`
    1. Note that `$DD_SCRATCH_DIR/$DD_OUTPUT_DIR` gets mounted at experiments/outputs which is the default output path!
3. Create whatever folders you need to create for logs, output, etc...


# Make params file 
```
./slurm/scripts/srun_make.sh
```
- Will ask you for the challenge, system, and model.
- Calls srun on dynadojo/slurm/jobscripts/make.sh
- Prints out the location of params.json (inside singularity container)

# Run params file
## Testing
Example of running a params file. The supplied path must be relative to inside the singularity container. 
```
./slurm/scripts/srun_submit.sh \
    slurm/jobscripts/sbatch/run.sbatch \
    experiments/outputs/fc/lds/fc_lds_lr_l=10/params.json \
    0,1
```
- Calls srun to...
- Run jobs 0 and 1 for fc/lds/fc_lds_lr_l=10/params.json
Unless you change how dynadojo/slurm/jobscripts/make.sh works or the CLI, usually the path the params will be the same as if you ran the CLI on your local computer with default output_dir. 

## Batch Jobs

### **Use the helper script!**
```
./slurm/scripts/sbatch_run.sh
```

### Do manually
```
./slurm/scripts/sbatch_submit.sh -J dynadojo_run --array=1-100 \
    slurm/jobscripts/sbatch/run.sbatch \
    experiments/outputs/fc/lds/fc_lds_lr_l=10/params.json \
    <optional-job-ids>
```
Calls sbatch with --array=1-100 with job name dynadojo_run. To monitor,
```
squeue -u <user>
```


# Make and Run (Together)
## Testing
Example of make and run in one command:
```
./slurm/scripts/srun_submit.sh slurm/jobscripts/sbatch/make_n_run.sbatch fc lds lr 0,1
```
- Calls srun to...
- Make params for fc lds lr
- Run jobs 0 and 1

## Batch Jobs
```
./slurm/scripts/sbatch_submit.sh -J dynadojo_mk_run --array=1-100 slurm/jobscripts/sbatch/make_n_run.sbatch fc lds lr <optional-job-ids>
```
Calls sbatch with --array=1-100 with job name dynadojo_mk_run. To monitor,
```
squeue -u <user>
```

# Check Jobs
```
./slurm/scripts/srun_check.sh 
```
- Will ask you the path to the data
- Path must be relative to the singularity container!
- For example, if the data is in `$DD_SCRATCH_DIR/$DD_OUTPUT_DIR\fc/lds/fc_lds_lr_5_l=5` then you should say `experiments/outputs/fc/lds/fc_lds_lr_5_l=5` because that is where it is mounted in the container

## Rerunning Missing Jobs
The list printed by check jobs can be supplied as an optional argument to run or makeNrun scripts. 

### Example Result of Running Check:
```
Loaded 560 rows from experiments/outputs/fc/lds/fc_lds_lr_5_l=5/fc_lds_lr_5_l=5.csv
Num of missing jobs:     75 of 100
Missing jobs: 
25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
```

### Example of Rerunning Missing Jobs:
```
./slurm/scripts/sbatch_submit.sh -J dynadojo_run --array=1-100 slurm/jobscripts/sbatch/run.sbatch experiments/outputs/fc/lds/fc_lds_lr_5_l=5/params.json 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
```

# Running the Experiments CLI interactively inside singularity container

## Example to plot
First, start an interactive singularity container:
```
./slurm/scripts/srun_interactive.sh
```
Then run whatever CLI commands you want, for example, to make plots:
```
python -u -m experiments plot --data_dir=experiments/outputs/fc/lds/fc_lds_lr_l\=10/
```

# Transfering files to your computer
Use Globus, or rsync. From your local computer:
```
export sunetid=<yourid>
rsync -avP ${sunetid}@login.sherlock.stanford.edu:/scratch/users/${sunetid}/sherput experiments/outputs/sherput
```

# Helpful Commands
Starting an interactive node (because you should not run anything on login nodes)
```
srun -c 1 --pty bash
```
Checking your jobs
```
squeue -u <user>
```
Canceling a job by id or name
```
scancel <job_id>
scancel -n <job-name>
```
Submitting a batch array job. (See dynadojo/slurm/jobscripts/sbatch for examples of jobs)
```
sbatch <sbatch_file> --array=1-100 -J <job-name>
```


