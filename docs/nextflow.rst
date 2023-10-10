.. _nextflow:

Nextflow
========

BayesTME provides a nextflow workflow for running a basic analysis on a Visium 10x dataset.

The only requirements for running the BayesTME nextflow pipeline locally are to install nextflow
(https://www.nextflow.io/docs/latest/getstarted.html) and docker (or singularity).

.. code::

    nextflow run main.nf -profile docker --input <path to spaceranger output> --n_cell_types 5 --outdir <output dir>


The results will be in the ``outdir`` directory specified in the params file, and will include
data and plots.

For more complicated workflows, see `./nextflow/modules` and `./nextflow/subworkflows` for composable components
that can be reused to author new pipelines.
