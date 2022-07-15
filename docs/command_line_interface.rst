Command Line Interface
======================

BayesTME provides a suite of command line utilities that allow users to script running the pipeline end to end on their platform.

These commands will be available on the path in the python environment in which the ``bayestme`` package is installed.

Loading Data
------------

We provide several utilities to convert data into format that BayesTME uses internally:

``load_spaceranger``

.. code::

    usage: load_spaceranger [-h] [--output OUTPUT] [--input INPUT]

    Convert data from spaceranger to a SpatialExpressionDataset in h5 format

    optional arguments:
      -h, --help       show this help message and exit
      --output OUTPUT  Output file, a SpatialExpressionDataset in h5 format
      --input INPUT    Input spaceranger dir



Gene Filtering
--------------

``filter_genes``

This command will create a new SpatialExpressionDataset that has genes
filtered according to adjustable criteria. One or more of the criteria can be specified.

.. code::

    usage: filter_genes [-h] [--output OUTPUT] [--input INPUT] [--filter-ribosomal-genes] [--n-top-by-standard-deviation N_TOP_BY_STANDARD_DEVIATION]
                        [--spot-threshold SPOT_THRESHOLD]

    Filter the genes based on one or more criteria

    optional arguments:
      -h, --help            show this help message and exit
      --output OUTPUT       Output file, a SpatialExpressionDataset in h5 format
      --input INPUT         Input SpatialExpressionDataset in h5 format
      --filter-ribosomal-genes
                            Filter ribosomal genes (based on gene name regex)
      --n-top-by-standard-deviation N_TOP_BY_STANDARD_DEVIATION
                            Use the top N genes with the highest spatial variance.
      --spot-threshold SPOT_THRESHOLD
                            Filter genes appearing in greater than the provided threshold of tissue spots.

Bleeding Correction
-------------------

``bleeding_correction``

.. code::

    usage: bleeding_correction [-h] [--input INPUT] [--bleed-out BLEED_OUT] [--stdata-out STDATA_OUT] [--n-top N_TOP] [--max-steps MAX_STEPS]
                               [--local-weight LOCAL_WEIGHT]

    Filter data

    optional arguments:
      -h, --help            show this help message and exit
      --input INPUT         Input file, SpatialExpressionDataset in h5 format
      --bleed-out BLEED_OUT
                            Output file, BleedCorrectionResult in h5 format
      --stdata-out STDATA_OUT
                            Output file, SpatialExpressionDataset in h5 format
      --n-top N_TOP         Use N top genes by standard deviation to calculate the bleeding functions. Genes will not be filtered from output dataset.
      --max-steps MAX_STEPS
                            Number of EM steps
      --local-weight LOCAL_WEIGHT
                            Initial value for local weight, a tuning parameter for bleed correction. rho_0g from equation 1 in the paper. By default will be set
                            to sqrt(N tissue spots)

Phenotype Selection
-------------------

``phenotype_selection``

.. code::

    usage: phenotype_selection [-h] [--stdata STDATA] [--job-index JOB_INDEX] [--n-fold N_FOLD] [--n-splits N_SPLITS] [--n-samples N_SAMPLES] [--n-burn N_BURN]
                               [--n-thin N_THIN] [--n-gene N_GENE] [--n-components-min N_COMPONENTS_MIN] [--n-components-max N_COMPONENTS_MAX]
                               [--lambda-values LAMBDA_VALUES] [--max-ncell MAX_NCELL] [--background-noise] [--lda-initialization] [--output-dir OUTPUT_DIR]

    Select values for number of cell types and lambda smoothing parameter via k-fold cross-validation.

    optional arguments:
      -h, --help            show this help message and exit
      --stdata STDATA       Input file, SpatialExpressionDataset in h5 format
      --job-index JOB_INDEX
                            Run only this job index, suitable for running the sampling in parallel across many machines
      --n-fold N_FOLD       Number of times to run k-fold cross-validation.
      --n-splits N_SPLITS   Split dataset into k consecutive folds for each instance of k-fold cross-validation
      --n-samples N_SAMPLES
                            Number of samples from the posterior distribution.
      --n-burn N_BURN       Number of burn-in samples
      --n-thin N_THIN       Thinning factor for sampling
      --n-gene N_GENE       Use N top genes by standard deviation to model deconvolution. If this number is less than the total number of genes the top N by
                            spatial variance will be selected
      --n-components-min N_COMPONENTS_MIN
                            Minimum number of cell types to try.
      --n-components-max N_COMPONENTS_MAX
                            Maximum number of cell types to try.
      --lambda-values LAMBDA_VALUES
                            Potential values of the lambda smoothing parameter to try. Defaults to (1, 1e1, 1e2, 1e3, 1e4, 1e5)
      --max-ncell MAX_NCELL
                            Maximum cell count within a spot to model.
      --background-noise
      --lda-initialization
      --output-dir OUTPUT_DIR
                            Output directory. N new files will be saved in this directory, where N is the number of cross-validation jobs.


Deconvolution
-------------

``deconvolve``

.. code::

    usage: deconvolve [-h] [--input INPUT] [--output OUTPUT] [--n-gene N_GENE] [--n-components N_COMPONENTS] [--lam2 LAM2] [--n-samples N_SAMPLES]
                      [--n-burnin N_BURNIN] [--n-thin N_THIN] [--random-seed RANDOM_SEED] [--bkg] [--lda]

    Deconvolve data

    optional arguments:
      -h, --help            show this help message and exit
      --input INPUT         Input SpatialExpressionDataset in h5 format, expected to be bleed corrected
      --output OUTPUT       Path where DeconvolutionResult will be written h5 format
      --n-gene N_GENE       number of genes
      --n-components N_COMPONENTS
                            Number of cell types, expected to be determined from cross validation.
      --lam2 LAM2           Smoothness parameter, this tuning parameter expected to be determinedfrom cross validation.
      --n-samples N_SAMPLES
                            Number of samples from the posterior distribution.
      --n-burnin N_BURNIN   Number of burn-in samples
      --n-thin N_THIN       Thinning factor for sampling
      --random-seed RANDOM_SEED
                            Random seed
      --bkg                 Turn background noise on
      --lda                 Turn LDA Initialization on


Spatial Differential Expression
-------------------------------

``spatial_expression``

.. code::

    usage: spatial_expression [-h] [--deconvolve-results DECONVOLVE_RESULTS] [--dataset DATASET] [--output OUTPUT] [--n-cell-min N_CELL_MIN]
                              [--n-spatial-patterns N_SPATIAL_PATTERNS] [--n-samples N_SAMPLES] [--n-burn N_BURN] [--n-thin N_THIN] [--simple] [--alpha0 ALPHA0]
                              [--prior-var PRIOR_VAR] [--lam2 LAM2] [--n-gene N_GENE]

    Detect spatial differential expression patterns

    optional arguments:
      -h, --help            show this help message and exit
      --deconvolve-results DECONVOLVE_RESULTS
                            DeconvolutionResult in h5 format
      --dataset DATASET     SpatialExpressionDataset in h5 format
      --output OUTPUT       Path to store SpatialDifferentialExpressionResult in h5 format
      --n-cell-min N_CELL_MIN
                            Only consider spots where there are at least <n_cell_min> cells of a given type, as determined by the deconvolution results.
      --n-spatial-patterns N_SPATIAL_PATTERNS
                            Number of spatial patterns.
      --n-samples N_SAMPLES
                            Number of samples from the posterior distribution.
      --n-burn N_BURN       Number of burn-in samples
      --n-thin N_THIN       Thinning factor for sampling
      --simple              Simpler model for sampling spatial differential expression posterior
      --alpha0 ALPHA0       Alpha0 tuning parameter. Defaults to 10
      --prior-var PRIOR_VAR
                            Prior var tuning parameter. Defaults to 100.0
      --lam2 LAM2           Smoothness parameter, this tuning parameter expected to be determined from cross validation.
      --n-gene N_GENE       Number of genes to consider for detecting spatial programs, if this number is less than the total number of genes the top N by
                            spatial variance will be selected


Plotting
--------

Creating plots is separated into separate commands:


``plot_bleeding``

.. code::

    usage: plot_bleeding [-h] [--raw-stdata RAW_STDATA] [--corrected-stdata CORRECTED_STDATA] [--bleed-correction-results BLEED_CORRECTION_RESULTS]
                         [--output-dir OUTPUT_DIR] [--n-top N_TOP]

    Plot bleeding correction results

    optional arguments:
      -h, --help            show this help message and exit
      --raw-stdata RAW_STDATA
                            Input file, SpatialExpressionDataset in h5 format
      --corrected-stdata CORRECTED_STDATA
                            Input file, SpatialExpressionDataset in h5 format
      --bleed-correction-results BLEED_CORRECTION_RESULTS
                            Input file, BleedCorrectionResult in h5 format
      --output-dir OUTPUT_DIR
                            Output directory
      --n-top N_TOP         Plot top n genes by stddev


``plot_deconvolution``

.. code::

    usage: plot_deconvolution [-h] [--stdata STDATA] [--deconvolution-result DECONVOLUTION_RESULT] [--output-dir OUTPUT_DIR] [--n-marker-genes N_MARKER_GENES]
                              [--alpha ALPHA] [--marker-gene-method {MarkerGeneMethod.TIGHT,MarkerGeneMethod.FALSE_DISCOVERY_RATE}]

    Plot deconvolution results

    optional arguments:
      -h, --help            show this help message and exit
      --stdata STDATA       Input file, SpatialExpressionDataset in h5 format
      --deconvolution-result DECONVOLUTION_RESULT
                            Input file, DeconvolutionResult in h5 format
      --output-dir OUTPUT_DIR
                            Output directory.
      --n-marker-genes N_MARKER_GENES
                            Plot top N marker genes.
      --alpha ALPHA         Alpha cutoff for choosing marker genes.
      --marker-gene-method {MarkerGeneMethod.TIGHT,MarkerGeneMethod.FALSE_DISCOVERY_RATE}
                            Method for choosing marker genes.

``plot_spatial_expression``

.. code::

    usage: plot_spatial_expression [-h] [--stdata STDATA] [--deconvolution-result DECONVOLUTION_RESULT] [--sde-result SDE_RESULT] [--output-dir OUTPUT_DIR]

    Plot deconvolution results

    optional arguments:
      -h, --help            show this help message and exit
      --stdata STDATA       Input file, SpatialExpressionDataset in h5 format
      --deconvolution-result DECONVOLUTION_RESULT
                            Input file, DeconvolutionResult in h5 format
      --sde-result SDE_RESULT
                            Input file, SpatialDifferentialExpressionResult in h5 format
      --output-dir OUTPUT_DIR
                            Output directory