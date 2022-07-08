#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

requirements:
 - class: InitialWorkDirRequirement
   listing: $(inputs.data_dir.listing)

inputs:
  - id: singularity_image
    type: File
  - id: data_dir
    type: Directory
  - id: n_gene
    type: string
  - id: lam2
    type: string
  - id: n_samples
    type: string
  - id: n_burnin
    type: string
  - id: n_thin
    type: string

arguments:
  - /opt/local/singularity/3.7.1/bin/singularity
  - exec
  - "--bind"
  - $(inputs.data_dir.path):/data
  - $(inputs.singularity_image.path)
  - deconvolve
  - "--data-dir"
  - /data
  - "--cv-likelihoods"
  - /data/k_fold/likelihoods
  - "--n-gene"
  - $(inputs.n_gene)
  - "--lam2"
  - $(inputs.lam2)
  - "--n-samples"
  - $(inputs.n_samples)
  - "--n-burnin"
  - $(inputs.n_burnin)
  - "--n-thin"
  - $(inputs.n_thin)

outputs: []