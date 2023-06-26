.. _nextflow:

Nextflow
========

BayesTME provides a nextflow workflow for running the entire pipeline

The only requirements for running the BayesTME nextflow pipeline locally are to install nextflow
(https://www.nextflow.io/docs/latest/getstarted.html) and docker.

The parameters template is defined in ``nextflow/nextflow.config``.

You can create a yaml file that defines the parameters for your run. This yaml file might look like this:

.. code::

    spaceranger_dir: /path/to/spaceranger/outs
    outdir: /path/to/results
    seed: 42
    inference_type: "SVI"


Once you have your parameters file ready, you can execute the pipeline with the following
command:

.. code::

    nextflow run https://github.com/tansey-lab/bayestme -r main -profile local -params-file '<path to params yaml>'


The results will be in the ``outdir`` directory specified in the params file, and will include raw
data and plots.
