# ------------------------------------------------------------------------------------#
######### BUILDER IMAGE #########
# Used to build runtime images. 
# Install dependencies and copy them to the runtime image.

FROM python:3.10-slim-buster as builder

RUN apt update -y \
    && apt-get update \
    && apt-get install -y gcc python3-dev build-essential python3-pkgconfig libopenblas-dev\
    && pip install -U pip setuptools wheel \
    && pip install pdm \
    touch README.md && touch LICENSE

# copy pyproject.toml (with dependencies)
COPY pyproject.toml /dynadojo/

# install dependencies in project
WORKDIR /dynadojo

#isolate dynadojo from dependencies & install from lockfile
RUN pdm config python.use_venv false && pdm install --prod -G tensorflow --no-lock --no-editable  --no-self


# ------------------------------------------------------------------------------------#
######### STANFORD SHERLOCK RUNTIME IMAGE #########
# Assumes that /home/users/ and /scratch are mounted and the dynadojo git repo is cloned in /home/users/$USER/dynadojo
# Build image with: 
#       docker build --target=sherlock --tag=dynadojo:sherlock .
# To build for multiple arch: 
#       Load and test locally:  docker buildx build --target=sherlock --platform=linux/amd64,linux/arm64 --tag=dynadojo:sherlock --load .
#       Push to dockerhub:      docker buildx build --target=sherlock --platform=linux/amd64,linux/arm64 --tag=carynbear/dynadojo:sherlock --push .
# To run on sherlock:
#       srun -c 4 --pty bash                    # request interactive session
#       mkdir -p $GROUP_HOME/$USER/simg         # make directory for singularity images
#       cd $GROUP_HOME/$USER/simg
#       singularity build --sandbox -F dynadojo docker://carynbear/dynadojo:sherlock
#       singularity run  --bind $HOME/dynadojo/experiments:/dynadojo/experiments --bind $HOME/dynadojo/src/dynadojo:/dynadojo/pkgs/dynadojo --pwd /dynadojo /home/groups/boahen/mkanwal/simg/dynadojo python -m experiments --output_dir=$SCRATCH/
#
#       singularity shell --writable dynadojo  # to enter sandbox instead of running a command

FROM python:3.10-slim as sherlock

# make the symlinked directories
RUN  mkdir -p /home/users \
    && mkdir -p /scratch 
# && mkdir -p /share/software/modules && mkdir -p /share/software/user && mkdir -p /oak && mkdir -p /home/groups && mkdir -p /etc/localtime && mkdir -p /etc/hosts

# retrieve dependencies packages from build stage
ENV PYTHONPATH=/dynadojo/pkgs
COPY --from=builder /dynadojo/__pypackages__/3.10/lib /dynadojo/pkgs

# symlink experiments and src to repo in the home directory
RUN ln -s /home/users/$USER/dynadojo/experiments /dynadojo/experiments && ln -s /home/users/$USER/dynadojo/src/dynadojo /dynadojo/pkgs/dynadojo

WORKDIR /dynadojo

#disable GPU
ENV CUDA_VISIBLE_DEVICES=-1

# set command/entrypoint, adapt to fit your needs
CMD ["python", "-m", "experiments"]




# ------------------------------------------------------------------------------------#
######### GENERAL USE RUNTIME IMAGE #########
# Build image with: 
#       docker build --target=runtime --tag=dynadojo .
# To build for multiple arch: 
#       Create buildx driver image:         docker buildx create --use
#       Build & Load to test locally:       docker buildx build --target=runtime --platform=linux/amd64,linux/arm64 --tag=dynadojo --load .
#       Or Build & Push to dockerhub:       docker buildx build --target=runtime --platform=linux/amd64,linux/arm64 --tag=[username]/dynadojo --push .


# run stage
FROM python:3.10-slim as runtime

# retrieve dependencies packages from build stage
ENV PYTHONPATH=/dynadojo/pkgs
COPY --from=builder /dynadojo/__pypackages__/3.10/lib /dynadojo/pkgs

COPY experiments/ /dynadojo/experiments

#isolate dynadojo from dependencies
COPY src/dynadojo /dynadojo/pkgs/dynadojo

WORKDIR /dynadojo

#disable GPU
ENV CUDA_VISIBLE_DEVICES=-1

# set command/entrypoint, adapt to fit your needs
CMD ["python", "-m", "experiments"]