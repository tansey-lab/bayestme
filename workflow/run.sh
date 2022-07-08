#!/bin/bash

module load java/jdk1.8.0_202

if [[ ! -f "cwlexec-0.2.2/cwlexec" ]]; then
  wget -Ocwlexec-0.2.2.tar.gz https://github.com/IBMSpectrumComputing/cwlexec/releases/download/v0.2.2/cwlexec-0.2.2.tar.gz

  tar -xzvf cwlexec-0.2.2.tar.gz
fi

# generate run id
run_id=$(date +"%Y-%m-%dT%H%M%S%z")

mkdir -p /work/tansey/${USER}/${run_id}

# run cwl workflow
cwlexec-0.2.2/cwlexec -w /work/tansey/${USER}/${run_id} \
  -o /work/tansey/${USER}/${run_id} \
  -c exec_config.json \
  bayestme.cwl \
  example.yaml