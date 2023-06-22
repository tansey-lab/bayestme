.. _nextflow:

Nextflow
========

BayesTME provides a nextflow workflow for running the entire pipeline

The parameters template is defined in ``nextflow/nextflow.config``.

There are many parameters for the model, but almost all of them are already set to reasonable defaults, so you
in practice should only need to set a few of them.

Your will define the parameters for your run by creating a yaml file,
a minimal example would look like this:

.. code::

    spaceranger_dir: /path/to/spaceranger/outs
    outdir: /path/to/results/dir
    inference_type: SVI
    seed: 42

You can then run the pipeline with the following command:

.. code::

    nextflow run https://github.com/tansey-lab/bayestme -r main -params-file '<path to params yaml>'


The results will be in the ``outdir`` directory specified in the params file, and will include raw
data and plots.
