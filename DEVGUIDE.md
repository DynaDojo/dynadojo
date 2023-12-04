# Recommended Tools
1. pyenv for managing python versions
2. pdm for managing virtual environments and packages

If you don't have these installed, you can install them with `homebrew` on MacOS. 

# Installing locally
## Python Version
Make sure you are working with a python version specified in pyproject.toml (3.10, 3.11)

If using pyenv,
```
pyenv versions #check installed versions
pyenv install 3.10
pyenv local 3.10 #set the current directory default version
```

## Package Installation
Optional packages are `tensorflow` and `rebound`.
```
pdm install -G tensorflow -G rebound
```
Reload terminal to activate the venv.

# Running Tests
```
pdm run python -m unittest
```

# Using Docker
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

# Advanced: Building/Pushing multiple OS/arch packages
1. First, Switch to different driver:  `docker buildx create --use`
2. Then, Build & Load to test locally: `docker buildx build --target=runtime --platform=linux/amd64,linux/arm64 --tag=dynadojo:test --load .`
3. Or Build & Push to dockerhub:       `docker buildx build --target=runtime --platform=linux/amd64,linux/arm64 --tag=[repository]/dynadojo:[version] --push .`
