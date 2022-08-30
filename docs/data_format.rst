Data Format
===========

Input Format
------------

The primary input format for BayesTME is `AnnData <https://anndata.readthedocs.io/en/latest/>`_

The input AnnData object is expected to have the following fields:

- ``adata.X`` - N spots x N markers integer matrix representing read counts.
- ``adata.obsm['spatial`]`` - spatial coordinates of the reads in the tissue slide.
- ``adata.obs['in_tissue`]`` - boolean array indicating whether a read comes from a tissue or non tissue spot.
- ``adata.uns['layout']`` - either SQUARE or HEX, corresponding to the probe layout geometry.
- ``adata.obsp['connectivities']`` - sparse boolean matrix indicating whether two observations neighbor each other in the probe grid or not.

This AnnData scheme is designed to be compatible with the scheme used by `scanpy <https://scanpy.readthedocs.io/en/stable/index.html>`_.

We have a provided a helper method, :py:meth:`bayestme.data.SpatialExpressionDataset.read_spaceranger`,
for creating the above AnnData object from raw `spaceranger <https://github.com/sbooeshaghi/spaceranger>`_ output.


Output Format
-------------

There are 4 high level data classes that represent the outputs of the 4 steps:

- :py:class:`bayestme.data.BleedCorrectionResult` - this is the output of bleeding correction
- :py:class:`bayestme.data.PhenotypeSelectionResult` - this is the output of one phenotype selection job in the larger grid search.
- :py:class:`bayestme.data.DeconvolutionResult` -  this is the output of sampling the deconvolution posterior distribution.
- :py:class:`bayestme.data.SpatialDifferentialExpressionResult` - this is the output of sampling the spatial differential expression posterior distribution.

These are all saved as `hdf5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ archives on disk, which is a format also used by AnnData.

:py:module:`bayestme.cli.bleeding_correction` produces two outputs, one is a modified version of the input
:py:class:`bayestme.data.SpatialExpressionDataset` object with the read counts adjusted,
the other is the :py:class:`bayestme.data.BleedCorrectionResult` object which contains the basis functions,
global, and local weights.

:py:module:`bayestme.cli.phenotype_selection` will produce one :py:class:`bayestme.data.PhenotypeSelectionResult` object
for each node in the grid search. Having one output per parameter set enables parallelization.

:py:module:`bayestme.cli.deconvolve` will produce a :py:class:`bayestme.data.DeconvolutionResult` object in h5 format,
which represents all the raw samples from the posterior distribution.
This object can be quite large (>10GB) as it is contains very high dimensional arrays of floating point numbers.
:py:module:`bayestme.cli.deconvolve` will also modify the AnnData archive to add meaningful summary statistics
from these posterior samples which are used in data visualization and analysis.

The AnnData fields added by this step are as follows:

Deconvolve AnnData Fields
^^^^^^^^^^^^^^^^^^^^^^^^^

- `adata.uns['bayestme_n_cell_types']` - integer, number of cell types
- `adata.varm['bayestme_cell_type_counts']` - <N marker> x <N cell type matrix> with the average posterior count of each
cell type in each spot
- `adata.varm['bayestme_cell_type_probabilities']` - <N marker> x <N cell type matrix> with the cell type probability of each
cell type in each spot


:py:module:`bayestme.cli.select_marker_genes` will modify the AnnData archive add to indicators of which genes are
marker genes for each cell type, and their order of significance.

The AnnData fields added by this step are as follows:

Marker Gene AnnData Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^

- `adata.varm['bayestme_cell_type_marker']` - <N marker> x <N cell type> integer matrix. Set to -1 if gene is not a
marker gene for cell type, otherwise set to monotonically increasing 0-indexed integers indicating marker gene
significance.
- `adata.varm['bayestme_omega_difference']` - <N marker> x <N cell type> floating point matrix. This statistic
represents the "overexpression" of a gene in a cell type, and is used for scaling the dot size in our marker gene plot.

:py:module:`bayestme.cli.spatial_expression` will produce a :py:class:`bayestme.data.SpatialDifferentialExpressionResult`
object in h5 format which represents all the raw samples from the posterior distribution.
This object can be quite large (>10GB) as it is contains very high dimensional arrays of floating point numbers.