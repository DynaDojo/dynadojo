# Notes

1. Connect to the cluster, preferably using VSCode remote connection
2. Figure out [MaxJobs](https://stackoverflow.com/a/61587377) & remember it
3. Make an `env.sh` file, see `env_example.sh`

Start an interactive node session: `srun -c 1 --pty bash`
Start an interactive singularity: 

# Helpful Commands

This [website](https://slurm.schedmd.com/job_array.html) is helpful

```bash
squeue -u <user-name>
scancel <job_id>
sbatch <sbatch_file> --array=1-100 -J <job-name>
scancel -n <job-name>
smanage report --sacct --name=dynadojo #get report
```
