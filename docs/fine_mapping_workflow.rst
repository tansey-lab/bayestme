Fine Mapping Workflow
=====================

BayesTME can take advantage of companion scRNA data.

Instead of using cross validation on the ST dataset to determine the cell types,
cell types can be determined from the companion scRNA directly, using an established
workflow such as `Seurat <https://satijalab.org/seurat>`_.

The resulting relative expression values of each gene in each cell type/cluster
(represented by φ_kg in equation 4 in the BayesTME preprint) can be provided to the deconvolution step
via the ``--expression-truth`` option (CLI) or ``expression_truth=`` parameter in
:py:meth:`bayestme.deconvolution.deconvolve`.

We have provided a docker container and script for running Seurat on 10Xgenomics/cellranger output,
which will produce the appropriate relative expression values in a CSV output that can
be read into the BayesTME pipeline.

.. code::

    docker run jeffquinnmsk/bayestme-seurat-fine-mapping:latest Rscript /