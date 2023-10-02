process DECONVOLVE {
    label 'process_high_memory'
    label 'process_long'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    publishDir "${params.outdir}/${sample_name}"

    input:
        path adata
        val sample_name
        val n_components


    output:
        path 'dataset_deconvolved.h5ad', emit: adata_output
        path 'deconvolution_samples.h5', emit: samples
        path 'loss.pdf', emit: loss_plot, optional: true

    script:
    def n_samples_flag = "--n-samples ${params.bayestme_deconvolution_n_samples}"
    def n_burn_flag = "--n-burn ${params.bayestme_deconvolution_n_burn}"
    def n_thin_flag = "--n-thin ${params.bayestme_deconvolution_n_thin}"
    def inference_type_flag = "--inference-type ${params.bayestme_inference_type}"
    def lda_initialization_flag = params.bayestme_lda_initialization ? "--lda-initialization" : ""
    def background_noise_flag = params.bayestme_background_noise ? "--background-noise" : ""
    def use_spatial_guide_flag = use_spatial_guide ? "--use-spatial-guide" : ""
    """
    deconvolve --adata ${adata} \
        --adata-output dataset_deconvolved.h5ad \
        --output deconvolution_samples.h5 \
        --spatial-smoothing-parameter ${spatial_smoothing_parameter} \
        --n-components ${n_components} \
        ${n_samples_flag} \
        ${n_burn_flag} \
        ${n_thin_flag} \
        ${use_spatial_guide_flag} \
        ${background_noise_flag} \
        ${lda_initialization_flag} \
        ${inference_type_flag}
    """
}

process SELECT_MARKER_GENES {
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    publishDir "${params.outdir}/${sample_name}"

    input:
        path adata
        path deconvolution_samples
        val sample_name

    output:
        path 'dataset_deconvolved_marker_genes.h5ad', emit: result
        path '*.csv', emit: csvs

    script:
    """
    select_marker_genes --adata ${adata} \
        --adata-output dataset_deconvolved_marker_genes.h5ad \
        --deconvolution-result ${params.bayestme_deconvolution_samples} \
        --n-marker-genes ${params.bayestme_n_marker_genes} \
        --alpha ${params.bayestme_marker_gene_alpha_cutoff} \
        --marker-gene-method ${params.bayestme_marker_gene_method}
    """
}

process PLOT_DECONVOLUTION {
    label 'process_single'
    publishDir "${params.outdir}/${sample_name}/plots/deconvolution"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
        path adata
        val sample_name

    output:
        path '*.pdf', emit: deconvolution_plots

    script:
    """
    plot_deconvolution --adata ${adata} \
        --output-dir .
    """
}

workflow DECONVOLUTION {
    take:
    adata
    sample_name
    n_components

    main:
    DECONVOLVE (adata,
            sample_name,
            n_components)

    SELECT_MARKER_GENES (
            DECONVOLVE.out.adata_output,
            DECONVOLVE.out.samples,
            sample_name)

    PLOT_DECONVOLUTION (
            SELECT_MARKER_GENES.out.result,
            sample_name)

    emit:
    plots = PLOT_DECONVOLUTION.out.deconvolution_plots
    adata = SELECT_MARKER_GENES.out.result
    samples = DECONVOLVE.out.samples
    marker_genes = SELECT_MARKER_GENES.out.csvs
}
