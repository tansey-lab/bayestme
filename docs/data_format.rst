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

Bleeding correction produces two outputs, one is a modified version of the input
:py:class:`bayestme.data.SpatialExpressionDataset` object with the read counts adjusted,
the other is the :py:class:`bayestme.data.BleedCorrectionResult` object which contains the basis functions,
global, and local weights.

Phenotype selection will produce one :py:class:`bayestme.data.PhenotypeSelectionResult` object for each node in the grid search.
Having one output per parameter set enables parallelization.

Deconvolution and spatial differential expression each just product their respective single objects as output,
these objects just represent the samples from the posterior distributions.