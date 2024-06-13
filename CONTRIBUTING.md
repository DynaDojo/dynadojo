# Contributing to DynaDojo

## Getting Started
### Prerequisites
- [Python](https://www.python.org/) (v3.10 or greater). We recommend using `mise` or `pyenv` for Python version management.
- [PDM](https://pdm-project.org/en/latest/) (v.2.15.4) for package management and for python environments.

### Additional System Requirements
**Optional Package**: The project can optionally depend on TensorFlow 2 for certain functionalities. Please refer to [Tensorflow 2 install](https://www.tensorflow.org/install) for any system requirements.

### Recommended Setup
0. For MacOS:
    1. **Install xcode tools**: xcode-select --install
    2. **Install homebrew**: Follow the instructions on [brew](https://brew.sh/)
1. **Install mise/pyenv**: Follow the instructions on [mise website](https://mise.jdx.dev/getting-started.html) or [pyenv GitHub page](https://github.com/pyenv/pyenv#installation).
    On macOS:

    1. **mise:** 
       ```sh
       brew install mise
       echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
       ```

    2. **pyenv:** 
       See steps 1-5 [here](https://ericsysmin.com/2024/02/05/how-to-install-pyenv-on-macos/)

3. **Install Python 3.10**

    1. **using mise:**
        ```sh
        mise use python@3.10.14
        ```       
    2. **using pyenv:** 
       ```sh
       pyenv install 3.10.14
       pyenv shell 3.10.14   #use the newly installed version
       ```

   Check your installation by running `which python` which should output something along the lines of:
   - for mise: `/Users/[user]/.local/share/mise/installs/python/3.10.14/bin/python`
   
   - for pyenv: `/Users/[user]/.pyenv/shims/python`
   
    If there are issues with your installation, check your `echo $PATH` variable for any other python installations. Remove them from `~/.zshrc` and `~/.zprofile` for ZSH (or `~/.bashrc` and `~/.bash_profile`) for BASH. 
   
5. **Install pdm with brew:**
   ```sh
   brew install pdm
   ```
6. **Fork the Repository:** Fork the repository on GitHub by clicking the "Fork" button on the repository's page. This creates a copy of the code under your GitHub user account
7. **Clone your forked repository:**
   ```sh
   git clone https://github.com/your-username/dynadojo.git
   cd dynadojo
8. **Add the Dynadojo upstream remote to your local Dynadojo clone:**
   ```sh
   git remote add upstream https://github.com/DynaDojo/dynadojo.git
   ```
9. **Configure git to pull from the upstream remote:**
   ```sh
   git switch main # ensure you're on the main branch
   git fetch upstream --tags
   git branch --set-upstream-to=upstream/main
   ```
10. **Set python version:**

    1. **mise**
        ```sh
        echo 3.10.14 > .python-version
        ```
        
    2. **pyenv**
       ```sh
       pyenv local 3.10.14 # set the default python version in current folder
       ```
11. **Install Dynadojo dependencies:**
    ```sh
    pdm install -G all
    ``` 
    If installation fails, please delete `pdm.lock` and try again. 

    1. **Optional: Install additional optional dependencies:**
       ```sh
       pdm add -G [optional package]
       ```
       For Macs with Apple Silicon, you might want to add `tensorflow-mac` for Mac GPU support when running tensorflow. Please check [Apple](https://developer.apple.com/metal/tensorflow-plugin/) for system requirements.

12. **Reload your terminal to activate the pdm venv.**
   or run
   ```shell
   $(pdm venv activate)
   ```
13. **Check your python path**:
   ```shell
   which python # should be [path to project]/DynaDojo/.venv/bin/python
   ```

## Making Changes

1. Make sure you're on the main branch.

   ```bash
   git switch main
   ```

2. Use the git pull command to retrieve content from the DynaDojo Github repository.

   ```bash
   git pull
   ```
3. Create a new branch and switch to it.

   ```bash
   git switch -c a-descriptive-name-for-my-changes
   ```
   Please use branch naming conventions. See [branch naming](#branch-naming) section below.
   
5. Make your changes!

6. Use the git add command to save the state of files you have changed.

   ```bash
   git add <names of the files you have changed>
   ```

7. Commit your changes.

   ```bash
   git commit
   ```
   Please remember to write [good commit messages](https://cbea.ms/git-commit/)
   
8. Rebase on upstream/main to keep your fork of the code up to date with the original repository.
   
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
   Resolve any conflicts, test your code, commit.
   
9. Push all changes to your fork of dynadojo (origin) on GitHub.
   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```
   You may have to push with the `--force` flag.

10. Click on Pull Request on Github to open a pull request. Make sure you tick off all the boxes on our [checklist](#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review. Once approved please squash commits and merge.

### Branch Naming
Please follow the convention of `[prefix]`/`[description]`/`[optional issue #]`/`[optional name]``. 
So for example `feat/transformer/caryn` or `bug/plotting/100`

#### Guidelines:
- Lowercase and Hyphen-separated: Stick to lowercase for branch names and use hyphens to separate words.
    - For instance, feature/new-login or bugfix/header-styling.
- Alphanumeric Characters: Use only alphanumeric characters (a-z, 0–9) and hyphens. Avoid punctuation, spaces, underscores, or any non-alphanumeric character.
- No Continuous Hyphens: Do not use continuous hyphens. feature--new-login can be confusing and hard to read.
- No Trailing Hyphens: Do not end your branch name with a hyphen. For example, feature-new-login- is not a good practice.
- Descriptive: The name should be descriptive and concise, ideally reflecting the work done on the branch.

#### Branch Prefixes
Using prefixes in branch names helps to quickly identify the purpose of the branches. Here are some types of branches with their corresponding prefixes:

- Feature Branches: These branches are used for developing new features.
    - Use the prefix feat/. For instance, feat/login-system.
- Bug fix Branches: These branches are used to fix bugs in the code.
    - Use the prefix bug/. For example, bug/header-styling.
- Release Branches: These branches are used to prepare for a new production release. They allow for last-minute dotting of i’s and crossing t’s.
    - Use the prefix release/. For example, release/v1.0.1.
- Documentation Branches: These branches are used to write, update, or fix documentation.
    - Use the prefix docs/. For instance, docs/api-endpoints.
- Experiment Branches: These branches are used for running experiments.
    - Use the prefix exp/. For instance, exp/neural-ode
- Works-in-Progress Branches: These branches are for projects that won't be completed any time soon.
    - Use the prefix wip/.
- Junk Branches: Throwaway branch.
    - Use the prefix junk/.

### Pull request checklist
- The pull request title should summarize your contribution and should start with one of the following prefixes:
    - feat: (new feature for the user, not a new feature for build script)
    - fix: (bug fix for the user, not a fix to a build script)
    - docs: (changes to the documentation)
    - style: (formatting, missing semicolons, etc; no production code change)
    - refactor: (refactoring production code, eg. renaming a variable)
    - perf: (code changes that improve performance)
    - test: (adding missing tests, refactoring tests; no production code change)
    - chore: (updating grunt tasks etc; no production code change)
    - build: (changes that affect the build system or external dependencies)
    - ci: (changes to configuration files and scripts)
    - revert: (reverts a previous commit)
    - **_Example_**: "feat: add support for PyTorch". 
    - **_Note_**: This is based on the [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/#summary)
- If your pull request addresses an issue, please mention the issue number in the pull
request description and title to make sure they are linked (and people viewing the issue know you
are working on it).
- To indicate a work in progress please prefix the title with `[WIP]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.
- Make sure existing tests pass.
- If adding a new feature, also add tests for it.
- All public methods must have informative docstrings.

## Running Tests
```
pdm run python -m unittest
```


<!-- Contributing.md Inspo:
https://raw.githubusercontent.com/bentoml/BentoML/main/DEVELOPMENT.md
https://raw.githubusercontent.com/huggingface/transformers/main/CONTRIBUTING.md
-->


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
