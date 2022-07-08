#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow

requirements:
  ScatterFeatureRequirement: { }
  SubworkflowFeatureRequirement: { }
  InlineJavascriptRequirement: { }
  StepInputExpressionRequirement: { }

inputs:
  - id: data_dir
    type: Directory
  - id: raw_data
    type: File
  - id: docker_image
    type: string
  - id: n_gene
    type: string
  - id: filter_type
    type: string
  - id: deconvolve_lam2
    type: string
  - id: deconvolve_n_samples
    type: string
  - id: deconvolve_n_burnin
    type: string
  - id: deconvolve_n_thin
    type: string
  - id: spatial_expression_n_samples
    type: string
  - id: spatial_expression_n_burnin
    type: string
  - id: spatial_expression_n_thin
    type: string
  - id: n_spatial_patterns
    type: string
outputs: []

steps:
  prepare_image:
    run: prepare_image.cwl
    in:
      docker_image: docker_image
    out:
      - singularity_image
  filter_bleed:
    run: filter_bleeding_correction.cwl
    in:
      data_dir: data_dir
      raw_data: raw_data
      singularity_image: prepare_image/singularity_image
      n_gene: n_gene
      filter_type: filter_type
    out: []
  prepare_kfold:
    run: prepare_kfold.cwl
    in:
      data_dir: data_dir
      raw_data: raw_data
      singularity_image: prepare_image/singularity_image
    out: [ cross_validation_configs ]
  run_kfold:
    run: run_kfold.cwl
    scatter: cross_validation_config
    in:
      data_dir: data_dir
      singularity_image: prepare_image/singularity_image
      cross_validation_config: prepare_kfold/cross_validation_configs
    out: []
  deconvolve:
    run: deconvolve.cwl
    in:
      data_dir: data_dir
      singularity_image: prepare_image/singularity_image
      n_gene: n_gene
      lam2: deconvolve_lam2
      n_samples: deconvolve_n_samples
      n_burnin: deconvolve_n_burnin
      n_thin: deconvolve_n_thin
    out: []
  spatial_expression:
    run: spatial_expression.cwl
    in:
      singularity_image: prepare_image/singularity_image
      data_dir: data_dir
      n_samples: spatial_expression_n_samples
      n_burnin: spatial_expression_n_burnin
      n_thin: spatial_expression_n_thin
      n_spatial_patterns: n_spatial_patterns
    out: []