# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change. 

# System Requirements
For installing [Tensorflow 2](https://www.tensorflow.org/install) as of 4/26/2024,

- Ubuntu 16.04 or later
- Windows 7 or later (with C++ redistributable)
- macOS 10.12.6 (Sierra) or later (no GPU support)
- WSL2 via Windows 10 19044 or higher including GPUs (Experimental)


# Setting up your environment
## Recommended Tools
1. **for Macs**: xcode-select --install
2. **for Macs/Linux**: homebrew (for installing pyenv and pdm)
3. pyenv for managing python versions
4. pdm (version 2.15.1) for managing virtual environments and packages

## Installing Dependencies locally
### Python Version
Make sure you are working with a python version specified in pyproject.toml (3.10, 3.11)

If using pyenv,
```
pyenv versions #check installed versions
pyenv install 3.10
pyenv local 3.10 #set the current directory default version
```

### Package Installation
Optional packages are `tensorflow` or `tensorflow-m1`,  and `rebound`.
```
pdm install -d -G tensorflow[-m1] -G rebound
```

Apple Silicon can install `tensorflow-m1`,  see system requirements in https://developer.apple.com/metal/tensorflow-plugin/

Reload your terminal to activate the venv.

# Running Tests
```
pdm run python -m unittest
```

<!-- DOCKER INSTRUCTIONS (OUT OF DATE)

# Using Docker Image
## Get the image
```
docker pull ghcr.io/dynadojo/dynadojo:latest
```
## Or Build the image locally
Navigate to where the dockerfile is.
```
docker build --target=runtime --tag=dynadojo:test .
```
## Run the image locally
Run experiment cmdline: 
```
docker run dynadojo:test
```

Or interactively: 
```
docker run -it --entrypoint /bin/bash dynadojo:test
```

## Pushing to the Git container registry
Must be a member of the DynaDojo Git org to have access to the `ghcr.io/dynadojo/` repository. We don't have a DockerHub account yet. 

1. Generate a personal access token & login to ghrc.io registry. (see this [https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic])
2. Tag your built image:  
```
docker tag dynadojo:test ghcr.io/dynadojo/dynadojo:[version]
```
3. Push image:            
```
docker push ghcr.io/dynadojo/dynadojo:[version]
```

## Advanced: Building/Pushing multiple OS/arch packages
1. First, Switch to different driver:  `docker buildx create --use`
2. Then, Build & Load to test locally: `docker buildx build --target=runtime --platform=linux/amd64,linux/arm64 --tag=dynadojo:test --load .`
3. Or Build & Push to dockerhub:       `docker buildx build --target=runtime --platform=linux/amd64,linux/arm64 --tag=[repository]/dynadojo:[version] --push .` 

-->








<!--
!!!! OLD INSTRUCTIONS FOR Stanford's Slurm cluster using Singularity. 

# Building Image
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


# Run Interactively on Sherlock
## 1. Login to Sherlock and Run your container [help](https://www.sherlock.stanford.edu/docs/software/using/singularity/#singularity-on-sherlock) [help](https://vsoch.github.io/lessons/singularity-quickstart/)

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

## 2. OR Build and run a writable singularity container to test [help](https://wiki.ncsa.illinois.edu/display/ISL20/Containers)
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


# Issues
1. Torch and Tensorflow are locked to specific version, need to isolate from dynadojo
2. Separate suites of systems and models from benchmarking system

-->
