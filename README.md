# BayesTME: A reference-free Bayesian method for analyzing spatial transcriptomics data

This package implements BayesTME, a fully Bayesian method for analyzing ST data without needing single-cell RNA-seq (scRNA) reference data.

## Setup

To setup a python virtualenv with required dependencies to run or develop this code locally:

```commandline
make VE
source VE/bin/activate
```

## Tutorial

See the file `demo.ipynb` for a tutorial on preprocessing real data to remove technical error with the BayesTME anisotropic correction and generating the K folds for cross-validation. BayesTME uses cross-validation to select the number of cell types. We recommend you run each fold setting separately (e.g. using a compute cluster).

Additional demos on running the deconvolution and spatial transcriptional program code are coming soon!

See the file `pipeline.sh` for an example of running the melanoma dataset locally from the command line.

## Sample Datasets:

Zebrafish A: https://www.dropbox.com/sh/1nbaa3dxcgco6oh/AACUD6KJT7KFGD7y7XQ1ndz-a?dl=0
Zebrafish B: https://www.dropbox.com/sh/7x312y5c9rsphzf/AABmQEaxlo5Lf-6x8yR4rMt4a?dl=0
Melanoma: https://www.dropbox.com/s/aha4mcdrq12myfi/ST_mel1_rep2_counts.tsv?dl=0

