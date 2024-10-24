FROM pytorch/pytorch:latest

USER root

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip pip install ipython memray


RUN mkdir /app
COPY src/ /app/src/
COPY tox.ini pyproject.toml setup.py setup.cfg LICENSE.txt README.md /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install '.[dev,test]'

ENV PYTHONUNBUFFERED=1

WORKDIR /app
