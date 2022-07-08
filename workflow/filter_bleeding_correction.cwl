#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

requirements:
 - class: InitialWorkDirRequirement
   listing: $(inputs.data_dir.listing)

inputs:
  - id: singularity_image
    type: File
  - id: raw_data
    type: File
  - id: filter_type
    type: string
  - id: n_gene
    type: string
  - id: data_dir
    type: Directory

arguments:
  - /opt/local/singularity/3.7.1/bin/singularity
  - exec
  - "--bind"
  - $(inputs.data_dir.path):/data
  - "--bind"
  - $(inputs.raw_data.path):/input
  - $(inputs.singularity_image.path)
  - filter_bleed
  - "--count-mat"
  - /input
  - "--data-dir"
  - /data
  - "--n-gene"
  - $(inputs.n_gene)
  - "--filter-type"
  - $(inputs.filter_type)

outputs: []