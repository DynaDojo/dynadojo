This repo is set up to work on Stanford Sherlock clusters using Singularity.

# Run Interactively on Sherlock

## 1. Locally build docker image (using BuildKit), from the project root, where the `Dockerfile` is.
```
DOCKER_BUILDKIT=1 docker build --target=runtime -t <username>/dynadojo .
```

Test it locally.
```
docker run -it <username>/dynadojo bash
```

To push the image onto docker hub:
```
docker push <username>/dynadojo:latest
```

## 2. Or use BuildX to build and push for multi-architecture
Alternatively, for a multi-architecture docker images, you need to use the newer build tool. [help](https://blog.jaimyn.dev/how-to-build-multi-architecture-docker-images-on-an-m1-mac/). [help2](https://nielscautaerts.xyz/making-dockerfiles-architecture-independent.html#:~:text=The%20Dockerfile%20and%20docker%20buildx)

```
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 --push -t carynbear/dynadojo:buildx --progress=plain --target=runtime .
```

To test, you would need to pull the image from the hub since this creates a BuildX container to build the image. Alternatively, if you want to load the generated image into your docker you can use the flag `--load` instead of `--push`. 

## 3. Login to Sherlock and Run your container [help](https://www.sherlock.stanford.edu/docs/software/using/singularity/#singularity-on-sherlock) [help](https://vsoch.github.io/lessons/singularity-quickstart/)

Request an interactive node
```
srun -c 4 --pty bash
```
Pull the image.
```
mkdir -p $GROUP_HOME/$USER/simg
cd $GROUP_HOME/$USER/simg
singularity pull docker://<username>/dynadojo
```
Run from entrypoint.
```
singularity run --pwd /dynadojo dynadojo_latest.sif 
```
Or run a shell interactively in the container.
```
mkdir /tmp/dynadojo
singularity shell --pwd /tmp/dynadojo dynadojo.simg
```

## 4. OR Build and run a writable singularity container to test [help](https://wiki.ncsa.illinois.edu/display/ISL20/Containers)
```
cd $GROUP_HOME/$USER/simg
singularity build --sandbox -F dynadojo docker://carynbear/dynadojo:buildx
singularity shell --writable dynadojo

singularity run --pwd /dynadojo  dynadojo
```
# Resources

## Writing Multi-Stage Dockerfiles to improve your builds
- [Use PDM in a multi-stage Dockerfile](https://pdm.fming.dev/latest/usage/advanced/#use-pdm-in-a-multi-stage-dockerfile)
- [Blazing fast Python Docker builds with Poetry](https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0)

## Notes
- Using `pdm sync` instead of `install` because for some reason tensorflow dependencies will not install properly for the amd build. Hopefully installing from the lockfile will fix this. 
- Not installing the `dynadojo` project (`--no-self`) with the dependencies so that it's a separate layer that doesn't have to be updated all the time. According to [this](https://github.com/pdm-project/pdm/issues/444) it should prevent reinstalling with every update. If this doesn't work, use the mounted folders to manage the dynadojo code. 
- Issues with `torch` version [details](https://stackoverflow.com/questions/76327419/valueerror-libcublas-so-0-9-not-found-in-the-system-path)
- Issues with `tensorflow-io-gcs-filesystem` version 
- Maybe you want tensorflow to be faster [link](https://gist.github.com/grantstephens/74468679558950dc66714ff3d672a782)

## Singularity Mounts
WARNING: passwd file doesn't exist in container, not updating
WARNING: group file doesn't exist in container, not updating
WARNING: Skipping mount /lscratch [hostfs]: /lscratch doesn't exist in container
WARNING: Skipping mount /share/software/modules [hostfs]: /share/software/modules doesn't exist in container
WARNING: Skipping mount /share/software/user [hostfs]: /share/software/user doesn't exist in container
WARNING: Skipping mount /scratch [hostfs]: /scratch doesn't exist in container
WARNING: Skipping mount /home/users [hostfs]: /home/users doesn't exist in container
WARNING: Skipping mount /oak [hostfs]: /oak doesn't exist in container
WARNING: Skipping mount /home/groups [hostfs]: /home/groups doesn't exist in container
WARNING: Skipping mount /etc/localtime [binds]: /etc/localtime doesn't exist in container
WARNING: Skipping mount /etc/hosts [binds]: /etc/hosts doesn't exist in container
WARNING: Skipping mount /var/apptainer/mnt/session/etc/resolv.conf [files]: /etc/resolv.conf doesn't exist in container

# Issues
1. Torch and Tensorflow are locked to specific version, need to isolate from dynadojo
2. Separate suites of systems and models from benchmarking system