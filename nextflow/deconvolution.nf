def create_expression_truth_flag(expression_truth_values) {
    if (expression_truth_values == null || expression_truth_values.length == 0) {
        return ""
    } else {
        var expression_truth_flag = ""
        for (expression_truth_value in expression_truth_values) {
            expression_truth_flag += "--expression-truth ${expression_truth_value} "
        }

        return expression_truth_flag
    }
}

process DECONVOLVE {
    label 'process_high_memory'
    label 'process_long'

    publishDir "${params.outdir}/deconvolution"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
        path adata
        val n_components
        val spatial_smoothing_parameter
        val use_spatial_guide
        val inference_type

    output:
        path 'dataset_deconvolved.h5ad', emit: adata_output
        path 'deconvolution_samples.h5', emit: samples
        path 'loss.pdf', emit: loss_plot, optional: true

    script:
    def n_samples_flag = "--n-samples ${params.deconvolution_n_samples}"
    def n_burn_flag = "--n-burn ${params.deconvolution_n_burn}"
    def n_thin_flag = "--n-thin ${params.deconvolution_n_thin}"
    def inference_type_flag = "--inference-type ${params.inference_type}"
    def lda_initialization_flag = params.lda_initialization ? "--lda-initialization" : ""
    def background_noise_flag = params.background_noise ? "--background-noise" : ""
    def use_spatial_guide_flag = use_spatial_guide ? "--use-spatial-guide" : ""
    def expression_truth_flag = create_expression_truth_flag(params.expression_truth_files)
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
        ${expression_truth_flag} \
        ${inference_type_flag}
    """
}

process SELECT_MARKER_GENES {
    label 'process_single'
    publishDir "${params.outdir}/marker_gene_selection"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
        path adata
        path deconvolution_samples
        val n_marker_genes
        val marker_gene_alpha_cutoff
        val marker_gene_method

    output:
        path 'dataset_deconvolved_marker_genes.h5ad', emit: result
        path '*.csv', emit: csvs

    script:
    """
    select_marker_genes --adata ${adata} \
        --adata-output dataset_deconvolved_marker_genes.h5ad \
        --deconvolution-result ${deconvolution_samples} \
        --n-marker-genes ${n_marker_genes} \
        --alpha ${marker_gene_alpha_cutoff} \
        --marker-gene-method ${marker_gene_method}
    """
}

process PLOT_DECONVOLUTION {
    label 'process_single'

    publishDir "${params.outdir}/deconvolution_plots"

    input:
        path adata

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
        n_components
        spatial_smoothing_parameter
        n_marker_genes
        marker_gene_alpha_cutoff
        marker_gene_method
        use_spatial_guide
        inference_type

    main:
        DECONVOLVE (adata, n_components, spatial_smoothing_parameter, use_spatial_guide, inference_type)

        SELECT_MARKER_GENES (DECONVOLVE.out.adata_output, DECONVOLVE.out.samples, n_marker_genes, marker_gene_alpha_cutoff, marker_gene_method)

        PLOT_DECONVOLUTION (SELECT_MARKER_GENES.out.result)

    emit:
        plots = PLOT_DECONVOLUTION.out.deconvolution_plots
        adata = SELECT_MARKER_GENES.out.result
        samples = DECONVOLVE.out.samples
}
