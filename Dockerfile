# FROM python:3.10-slim-buster as builder
FROM tensorflow/tensorflow:2.14.0 as builder
ARG TARGETPLATFORM
RUN apt update -y \
    && apt-get install -y python3-dev build-essential \
    && pip install -U pip setuptools wheel \
    && pip install pdm \
    touch README.md && touch LICENSE

# copy files
COPY pyproject.toml pdm.lock /dynadojo/
COPY src/ /dynadojo/src
# COPY experiments/ /dynadojo/experiments

# install dependencies and project
WORKDIR /dynadojo

#isolate dynadojo from dependencies & install from lockfile
RUN pdm config python.use_venv false && pdm install --prod --no-lock --no-editable  --no-self

#otherwise if dynadojo is stable
# RUN pdm config python.use_venv false && pdm install --prod --no-lock --no-editable 

# run stage
FROM python:3.11-slim as runtime

RUN  mkdir -p /home/users \
    && mkdir -p /scratch 
# && mkdir -p /scratch && mkdir -p /share/software/modules && mkdir -p /share/software/user && mkdir -p /oak && mkdir -p /home/groups && mkdir -p /etc/localtime && mkdir -p /etc/hosts
# retrieve dependencies packages from build stage
ENV PYTHONPATH=/dynadojo/pkgs
COPY --from=builder /dynadojo/__pypackages__/3.10/lib /dynadojo/pkgs
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then pip install tensorflow==2.14.0 torch==2.0.0; fi
COPY experiments/ /dynadojo/experiments

#isolate dynadojo from dependencies 
COPY src/dynadojo /dynadojo/pkgs/dynadojo

WORKDIR /dynadojo

# set command/entrypoint, adapt to fit your needs
CMD ["python", "-m", "experiments"]