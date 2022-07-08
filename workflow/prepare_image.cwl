#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

inputs:
  - id: docker_image
    type: string

arguments:
  - /opt/local/singularity/3.7.1/bin/singularity
  - pull
  - $(inputs.docker_image)

outputs:
  singularity_image:
    type: File
    outputBinding:
      glob: "*.sif"