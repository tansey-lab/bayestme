#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

requirements:
  - class: InitialWorkDirRequirement
    listing: $(inputs.data_dir.listing)
  - class: InlineJavascriptRequirement

inputs:
  - id: singularity_image
    type: File
  - id: data_dir
    type: Directory
  - id: cross_validation_config
    type: File

arguments:
  - /opt/local/singularity/3.7.1/bin/singularity
  - exec
  - "--bind"
  - $(inputs.data_dir.path)/k_fold/jobs/data:/data
  - "--bind"
  - $(inputs.cross_validation_config.path):/config
  - $(inputs.singularity_image.path)
  - grid_search
  - "--data-dir"
  - /data
  - "--config"
  - /config
  - "--output-dir"
  - /data/k_fold

outputs:
  []