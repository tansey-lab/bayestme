#!/bin/bash

set -e -o pipefail -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $SCRIPT_DIR/..

pip install -e '.[dev,test]'

python -c "import bayestme.synthetic_data; bayestme.synthetic_data.generate_demo_dataset().save('demo.h5ad')"
input_file=$(pwd)/demo.h5ad

if [[ -d "mcmc_nextflow_test" ]]; then
  rm -r mcmc_nextflow_test
fi

if [[ -d "svi_nextflow_test" ]]; then
  rm -r svi_nextflow_test
fi

mkdir -p mcmc_nextflow_test
cd mcmc_nextflow_test

nextflow run ../main.nf \
  -config ../nextflow.config \
  -profile local \
  --input_adata $input_file \
  --spot_threshold 1.0 \
  --phenotype_selection_n_splits 4 \
  --phenotype_selection_n_fold 2 \
  --phenotype_selection_n_burn 10 \
  --phenotype_selection_n_thin 1 \
  --phenotype_selection_n_components_max 4 \
  --phenotype_selection_n_components_min 2 \
  --deconvolution_n_burn 10 \
  --n_marker_genes 3 \
  --marker_gene_alpha_cutoff 1 \
  --outdir ./results \
  --deconvolution_n_samples 100 \
  --deconvolution_n_thin 1 \
  --seed 1 \
  --spatial_expression_n_burn 10 \
  --spatial_expression_n_samples 100 \
  --spatial_expression_n_thin 1 \
  --significant_spatial_pattern_tissue_threshold 1 \
  --significant_spatial_pattern_moran_i_score_threshold 0.7 \
  --significant_spatial_pattern_gene_spatial_pattern_proportion_threshold 0.5 \
  --inference-type MCMC \
  -resume

cd ..
mkdir -p svi_nextflow_test
cd svi_nextflow_test

nextflow run ../main.nf \
  -config ../nextflow.config \
  -profile local \
  --input_adata $input_file \
  --spot_threshold 1.0 \
  --phenotype_selection_n_splits 4 \
  --phenotype_selection_n_fold 2 \
  --phenotype_selection_n_burn 10 \
  --phenotype_selection_n_thin 1 \
  --phenotype_selection_n_components_max 4 \
  --phenotype_selection_n_components_min 2 \
  --deconvolution_n_burn 10 \
  --n_marker_genes 3 \
  --marker_gene_alpha_cutoff 1 \
  --outdir ./results \
  --deconvolution_n_samples 100 \
  --deconvolution_n_thin 1 \
  --seed 1 \
  --spatial_expression_n_burn 10 \
  --spatial_expression_n_samples 100 \
  --spatial_expression_n_thin 1 \
  --significant_spatial_pattern_tissue_threshold 1 \
  --significant_spatial_pattern_moran_i_score_threshold 0.7 \
  --significant_spatial_pattern_gene_spatial_pattern_proportion_threshold 0.5 \
  --inference-type SVI \
  -resume
