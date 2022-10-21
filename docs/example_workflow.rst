.. _example-workflow:

Example Workflow
================

This purpose of this document is to give you a sense of how you can string the different BayesTME modules together
to analyze your data. This is a maximal example, you don't necessarily need to perform all of these steps,
but some of them do depend on the output of each other.

1. Assuming your data is the output of the visium/10x spaceranger pipeline, the first step is to convert that spaceranger
output into an anndata archive, which is the primary data format of BayesTME. You can do this with the :ref:`load_spaceranger <cli_load_spaceranger>`
command:

.. code::

    load_spaceranger --input spaceranger_input_dir --output dataset.h5ad


2. Filter the genes in your dataset using the :ref:`filter_genes <cli_filter_genes>` command

.. code::

    filter_genes --adata dataset.h5ad \
        --filter-ribosomal-genes \
        --n-top-by-standard-deviation 1000 \
        --output dataset_filtered.h5ad

This will create a new anndata archive ``dataset_filtered.h5ad`` which has only the selected genes in it.

3. Run bleed correction using the :ref:`bleeding_correction <cli_bleeding_correction>` command

.. code::

    bleeding_correction --adata dataset_filtered.h5ad \
        --adata-output dataset_filtered_corrected.h5ad \
        --bleed-out bleed_correction_results.h5

This will create a new anndata archive ``dataset_filtered_corrected.h5ad`` where the counts have been replaced with the
bleed corrected counts. This will also create another h5 archive, which is a serialized :py:class:`bayestme.data.BleedCorrectionResult`.

4. Plot bleeding correction using the :ref:`plot_bleeding_correction <cli_plot_bleeding_correction>` command
command:

.. code::

    plot_bleeding_correction --raw-adata dataset_filtered.h5ad \
        --corrected-adata dataset_filtered_corrected.h5ad \
        --bleed-correction-results bleed_correction_results.h5 \
        --output-dir bleed_correction_results


5. Run phenotype selection / cross validation using the :ref:`phenotype_selection <cli_bleeding_correction>`.
This step is very computationally expensive as we need to re-run the deconvolution gibbs sampler thousands of times
in order to do cross validation to learn the number of cell types and the lambda parameter.
This step cannot be feasibly accomplished on a single computer,
a computational cluster or cloud service provider needs to be used in order to run many MCMC samplers in parallel.

If you have some outside data telling you how many cell types are in your sample you can feasibly skip this step and go straight to step 6,
however you will need to have a reasonable guess for the lambda parameter. If you are taking this quick and dirty approach,
``lambda = 1000`` is probably a reasonable guess.

We cannot comment too much on how to set up this distributed computation, as it will vary a lot depending on whether you are using
AWS, Google cloud, or a high performance computing cluster, but it basically boils down to running the :ref:`phenotype_selection <cli_phenotype_selection>`
command N times in parallel:

.. code::

    phenotype_selection --adata dataset_filtered_corrected.h5ad \
        --output-dir phenotype_selection_results \
        --job-index ${JOB_INDEX}

The number of jobs that need to be run to complete the sweep of the parameter space will be equal to
(n_lambda parameters) * (n_folds) * (number of different cell types).

By default we recommend trying 2 to 12 cell types (so 10 different cell types), 5 folds,
and lambda values ``(1, 1e1, 1e2, 1e3, 1e4, 1e5)``. So this is 300 different jobs that will need to be run to finish the
full parameter sweep.

In your distributed computing framework you would set it so that 300 jobs running the above command are kicked off,
and that the variable JOB_INDEX is set to values 0..299 inclusive.

When all 300 jobs are complete the output directory will have 300 h5 archives in it,
containing serialized :py:class:`bayestme.data.PhenotypeSelectionResult` objects.

When this is complete you can use the utility function :py:func:`bayestme.cv_likelihoods.plot_likelihoods` to plot
the results and see which value of lambda and n_cell_types performed the best.

6. Run deconvolution using :ref:`deconvolve <cli_deconvolve>` command.

.. code::

    deconvolve --adata dataset_filtered_corrected.h5ad \
        --adata-output dataset_deconvolved.h5ad \
        --output deconvolution_samples.h5 \
        --lam2 <value of lambda learned from step 4> \
        --n-components <value of n cell types learned from step 4>

This will create a new anndata archive ``dataset_deconvolved.h5ad`` which has been updated to
include the summarized deconvolution results.
This will also create another h5 archive, which is a serialized :py:class:`bayestme.data.DeconvolutionResult`.
The serialized :py:class:`bayestme.data.DeconvolutionResult` can be very large (~ 10GB) as it saves all of the MCMC
samples, each of which are high dimensional numerical arrays.


7. Select marker genes using :ref:`select_marker_genes <cli_select_marker_genes>` command.

.. code::

    select_marker_genes --adata dataset_deconvolved.h5ad \
        --adata-output dataset_deconvolved_marker_genes.h5ad \
        --deconvolution-result deconvolution_samples.h5 \
        --n-marker-genes 5

8. Plot deconvolution using the :ref:`plot_deconvolution <cli_plot_deconvolution>`
command:

.. code::

    plot_deconvolution --adata dataset_deconvolved_marker_genes.h5ad \
        --output-dir deconvolution_plots

This will create a new anndata archive ``dataset_deconvolved_marker_genes.h5ad`` which has annotations added to
note the selected marker genes.

9. Run spatial differential expression using the :ref:`spatial_expression <cli_spatial_expression>` command:

.. code::

    spatial_expression --adata dataset_deconvolved_marker_genes.h5ad \
        --output sde_samples.h5

This will an h5 serialized :py:class:`bayestme.data.SpatialDifferentialExpressionResult`.
The serialized :py:class:`bayestme.data.SpatialDifferentialExpressionResult` can be very large (~ 10GB) as
it saves all of the MCMC samples, each of which are high dimensional numerical arrays.


10. Plot spatial differential expression using the :ref:`plot_spatial_expression <cli_plot_spatial_expression>`
command:

.. code::

    plot_spatial_expression --adata dataset_deconvolved_marker_genes.h5ad \
        --deconvolution-result deconvolution_samples.h5 \
        --sde-result sde_samples.h5 \
        --output-dir sde_plots
