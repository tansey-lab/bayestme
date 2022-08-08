.. _command-line-interface:

Command Line Interface
======================

BayesTME provides a suite of command line utilities that allow users to script running the pipeline end to end.

These commands will be available on the path in the python environment in which the ``bayestme`` package is installed.

``load_spaceranger``
--------------------

.. argparse::
   :ref: bayestme.cli.load_spaceranger.get_parser
   :prog: load_spaceranger

``filter_genes``
----------------

This command will create a new SpatialExpressionDataset that has genes
filtered according to adjustable criteria. One or more of the criteria can be specified.

.. argparse::
   :ref: bayestme.cli.filter_genes.get_parser
   :prog: filter_genes

``bleeding_correction``
-----------------------

.. argparse::
   :ref: bayestme.cli.bleeding_correction.get_parser
   :prog: bleeding_correction

``phenotype_selection``
-----------------------

.. argparse::
   :ref: bayestme.cli.phenotype_selection.get_parser
   :prog: phenotype_selection

``deconvolve``
--------------

.. argparse::
   :ref: bayestme.cli.deconvolve.get_parser
   :prog: deconvolve

``spatial_expression``
----------------------

.. argparse::
   :ref: bayestme.cli.spatial_expression.get_parser
   :prog: spatial_expression

Plotting
--------

Creating plots is separated into separate commands:


``plot_bleeding``
-----------------

.. argparse::
   :ref: bayestme.cli.plot_bleeding_correction.get_parser
   :prog: plot_bleeding_correction

``plot_deconvolution``
----------------------

.. argparse::
   :ref: bayestme.cli.plot_deconvolution.get_parser
   :prog: plot_deconvolution

``plot_spatial_expression``
---------------------------

.. argparse::
   :ref: bayestme.cli.plot_spatial_expression.get_parser
   :prog: plot_spatial_expression