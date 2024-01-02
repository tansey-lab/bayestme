FROM python:3.11

RUN --mount=type=cache,target=/root/.cache pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache pip install ipython

RUN mkdir /app
COPY src/ /app/src/
COPY tox.ini pyproject.toml setup.py setup.cfg LICENSE.txt README.md /app/

RUN --mount=type=cache,target=/root/.cache cd /app && pip install '.[dev,test]' --extra-index-url https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1

WORKDIR /app
