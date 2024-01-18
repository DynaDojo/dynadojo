# Notes

## Tips
- Use ondemand to connect via code server - [docs](https://www.sherlock.stanford.edu/docs/user-guide/ondemand/?h=templat#vs-code)
- The singularity/docker container only contains the python dependencies that are in the 'main' branch of the repo. We mount the dynadojo code to whereever you clone the repo so that any local changes to code are updated in the container. If you need to add any dependencies, you must create a new docker image and pull the new image. See experiments/README.md for how you can build a new image. 
- You can use ondemand to submit jobs via templates, once you are set up! (ToDo: make a video showing this)

## Steps to running on Sherlock
1. Clone dynadojo repo. The singularity container mounts to the repo so that code changes get propogated.
```
git clone https://github.com/DynaDojo/dynadojo.git
```
2. Make the results directory in Scratch
```
mkdir $SCRATCH/sherput
```
3. Make experiment params & run experiment
Make:
```
chmod +x dynadojo/experiments/sherlock/jobscripts/make.sh #make the script executable
srun --export=all  -c 1 dynadojo/experiments/sherlock/jobscripts/make.sh fc lds lr
```
AND run:
```
sbatch $HOME/dynadojo/experiments/sherlock/jobscripts/run.sbatch <path-to-params-see-prev-job-output> <optional-job-id-list>
```
OR all in one:
```
sbatch $HOME/dynadojo/experiments/sherlock/jobscripts/make_n_run.sbatch fc lds lr <optional-job-id-list>
```
4. Check if all jobs were run
```
source experiments/sherlock/scripts/session.sh #If on a login node
source experiments/sherlock/scripts/interactive.sh #to interactively run singularity container
python -m experiments check --data_dir experiments/outputs/<path-to-exp-output>
```
5. Re-run failed jobs
Repeat step 3 with `<optional-job-id-list>` outputted in step 4. 

6. Optionally, transfer output files to local computer 

7. Create plot
