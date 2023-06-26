FROM python:3.9

RUN apt-get update -y && apt-get install -y openjdk-17-jre-headless

RUN cd /usr/bin && curl -s https://get.nextflow.io | bash

RUN apt-get update -y && apt-get install -y libsuitesparse-dev

RUN --mount=type=cache,target=/root/.cache pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache pip install ipython

RUN mkdir /app
COPY src/ /app/src/

COPY tox.ini pyproject.toml setup.py setup.cfg LICENSE.txt README.md /app/

RUN --mount=type=cache,target=/root/.cache cd /app && pip install '.[dev,test]' --extra-index-url https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1

WORKDIR /app
