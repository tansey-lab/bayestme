from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="bayestme",
    version="0.1.0",
    description="A reference-free Bayesian method that discovers spatial transcriptional programs in the tissue microenvironment",
    url="https://github.com/tansey-lab/bayestme",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={  # Optional
        "console_scripts": [
            "grid_search=bayestme.grid_search_cfg:main",
            "filter_genes=bayestme.cli.filter_genes:main",
            "bleeding_correction=bayestme.cli.bleeding_correction:main",
            "deconvolve=bayestme.cli.deconvolve:main",
            "spatial_expression=bayestme.cli.spatial_expression:main",
            "load_spaceranger=bayestme.cli.load_spaceranger:main",
            "plot_bleeding=bayestme.cli.plot_bleeding_correction:main",
            "plot_deconvolution=bayestme.cli.plot_deconvolution:main",
            "plot_spatial_expression=bayestme.cli.plot_spatial_expression:main",
            "phenotype_selection=bayestme.cli.phenotype_selection:main"
        ],
    },
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy>=1.22.4",
        "seaborn>=0.11.2",
        "scipy>=1.7",
        "scikit-image>=0.19.2",
        "scikit-learn>=1",
        "pypolyagamma>=1.2.3",
        "matplotlib>=3.5",
        "autograd-minimize>=0.2.2",
        "scikit-sparse>=0.4.6",
        "torch>=1.11.0",
        "torchaudio>=0.11.0",
        "torchvision>=0.12.0",
        "statsmodels>=0.13.2",
        "anndata>=0.8.0",
        "h5py>=3.7.0",
        "esda>=2.4.1",
        "libpysal>=4.5.1"
    ],
    extras_require={
        "dev": [
            "check-manifest"
        ],
        "test": [
            "pytest",
            "tox"
        ],
    }
)
