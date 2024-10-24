# BayesTME: A unified statistical framework for spatial transcriptomics

![tests](https://github.com/tansey-lab/bayestme/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/bayestme/badge/?version=latest)](https://bayestme.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/511984802.svg)](https://zenodo.org/badge/latestdoi/511984802)

This package implements BayesTME, a fully Bayesian method for analyzing ST data without needing single-cell RNA-seq (scRNA) reference data.

## Documentation

### [bayestme.readthedocs.io](https://bayestme.readthedocs.io/en/latest/)

## Citation

If you use this code, please cite the [manuscript](https://doi.org/10.1016/j.cels.2023.06.003):

```
Zhang H, Hunter MV, Chou J, Quinn JF, Zhou M, White RM, Tansey W. BayesTME: An end-to-end method for multiscale spatial transcriptional profiling of the tissue microenvironment. Cell Syst. 2023 Jul 19;14(7):605-619.e7. doi: 10.1016/j.cels.2023.06.003. PMID: 37473731; PMCID: PMC10368078.
```

Bibtex citation:
```
@article{Zhang:2023aa,
	author = {Zhang, Haoran and Hunter, Miranda V and Chou, Jacqueline and Quinn, Jeffrey F and Zhou, Mingyuan and White, Richard M and Tansey, Wesley},
	journal = {Cell Syst},
	month = {Jul},
	number = {7},
	pages = {605--619},
	title = {BayesTME: An end-to-end method for multiscale spatial transcriptional profiling of the tissue microenvironment.},
	volume = {14},
	year = {2023}}

```

## Developer Setup

Please run `make install_precommit_hooks` from the root of the repository
to install the pre-commit hooks.

When you run any `git commit` command these pre-commit hooks will run and format any files that you changed in your commit.

Any unchanged files will not be formatted.

### Internal Contributions

When contributing to this repository, please use the feature branch workflow documented here: https://github.com/tansey-lab/wiki/blob/master/FEATURE_BRANCH_WORKFLOW.md
