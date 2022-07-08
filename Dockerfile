FROM python:3.9

RUN apt-get update -y && apt-get install -y libsuitesparse-dev

RUN mkdir /app

COPY requirements.txt /opt/requirements.txt

RUN pip3 install -r /opt/requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install cython
RUN pip3 install https://github.com/slinderman/pypolyagamma/archive/refs/tags/1.2.3.tar.gz

COPY src/ /app/src/

COPY tox.ini pyproject.toml setup.py setup.cfg LICENSE.txt README.md /app/

RUN cd /app && python setup.py install

ENV PYTHONUNBUFFERED=1

WORKDIR /app