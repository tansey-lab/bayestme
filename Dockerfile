FROM python:3.9

RUN apt-get update -y && apt-get install -y libsuitesparse-dev

COPY requirements.txt /opt/requirements.txt

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip  pip3 install -r /opt/requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir /app
COPY src/ /app/src/

COPY tox.ini pyproject.toml setup.py setup.cfg LICENSE.txt README.md /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install '.[dev,test]'


ENV PYTHONUNBUFFERED=1

WORKDIR /app
