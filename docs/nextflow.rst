.. _nextflow:

Nextflow
========

BayesTME provides a nextflow workflow for running the entire pipeline

The parameters template is defined in ``nextflow/nextflow.config``.

You can create a yaml file that defines the parameters for your run and execute the pipeline with the following
command:


.. code::

    nextflow run https://github.com/tansey-lab/bayestme -r main -params-file '<path to params yaml>'


The results will be in the ``outdir`` directory specified in the params file, and will include raw
data and plots.
