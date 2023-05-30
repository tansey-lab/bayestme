# BayesTME: A unified statistical framework for spatial transcriptomics

![tests](https://github.com/tansey-lab/bayestme/actions/workflows/python-unittest.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/bayestme/badge/?version=latest)](https://bayestme.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/511984802.svg)](https://zenodo.org/badge/latestdoi/511984802)

This package implements BayesTME, a fully Bayesian method for analyzing ST data without needing single-cell RNA-seq (scRNA) reference data.

## Documentation

### [bayestme.readthedocs.io](https://bayestme.readthedocs.io/en/latest/)

## Citation

If you use this code, please cite the [preprint](https://www.biorxiv.org/content/10.1101/2022.07.08.499377):

```
BayesTME: A unified statistical framework for spatial transcriptomics
H. Zhang, M. V. Hunter, J. Chou, J. F. Quinn, M. Zhou, R. White, and W. Tansey
bioRxiv 2022.07.08.499377.
```

Bibtex citation:
```
@article {Zhang2022.07.08.499377,
	author = {Zhang, Haoran and Hunter, Miranda V and Chou, Jacqueline and Quinn, Jeffrey F and Zhou, Mingyuan and White, Richard and Tansey, Wesley},
	title = {{BayesTME}: {A} unified statistical framework for spatial transcriptomics},
	year = {2022},
	doi = {10.1101/2022.07.08.499377},
	journal = {bioRxiv}
}
```

## Developer Setup

Please run `make install_precommit_hooks` from the root of the repository
to install the pre-commit hooks.

When you run any `git commit` command these pre-commit hooks will run and format any files that you changed in your commit.

Any unchanged files will not be formatted.

### Internal Contributions

When contributing to this repository, please use the feature branch workflow documented here: https://github.com/tansey-lab/wiki/blob/master/FEATURE_BRANCH_WORKFLOW.md
