Reference scRNA
===============

BayesTME can take advantage of reference scRNA data in anndata h5ad format.

This will set a prior on the expression profiles of cell types in the spatial data, it will
also consequently set the number of celltypes.

Provide your anndata archive to ``deconvolve`` like so:

.. code::

    deconvolve --expression-truth companion_scRNA.h5ad \
        --reference-scrna-celltype-column celltype \
        --reference-scrna-sample-column sample


``deconvolve`` will consider all the samples jointly to determine
a prior on celltype expression profiles.

We assume the values of ``--reference-scrna-celltype-column`` and
``--reference-scrna-sample-column`` are attributes of the ``obs`` table in your
anndata object.
