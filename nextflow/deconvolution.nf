process DECONVOLVE {
    label 'big_mem'


    input:
        path adata
        val n_components
        val lambda

    output:
        path 'dataset_deconvolved.h5ad', emit: adata_output
        path 'deconvolution_samples.h5', emit: samples

    script:
    def n_samples_flag = "--n-samples ${params.deconvolution_n_samples}"
    def n_burn_flag = "--n-burn ${params.deconvolution_n_burn}"
    def n_thin_flag = "--n-thin ${params.deconvolution_n_thin}"
    def n_gene_flag = "--n-gene ${params.deconvolution_n_gene}"
    def phenotype_selection_background_noise_flag = params.background_noise ? "--background-noise" : ""
    def phenotype_selection_lda_initialization_flag = params.lda_initialization ? "--lda-initialization" : ""
    def expression_truth_flag = create_expression_truth_flag(params.expression_truth_files)
    """
    deconvolve --adata ${adata} \
        --adata-output dataset_deconvolved.h5ad \
        --output deconvolution_samples.h5 \
        --lam2 ${lambda} \
        --n-components ${n_components} \
        ${n_samples_flag} \
        ${n_burn_flag} \
        ${n_thin_flag} \
        ${n_gene_flag} \
        ${phenotype_selection_background_noise_flag} \
        ${phenotype_selection_lda_initialization_flag} \
        ${expression_truth_flag}
    """
}

process SELECT_MARKER_GENES {
    label 'small_mem'


    input:
        path adata
        path deconvolution_samples
        val n_marker_genes
        val marker_gene_alpha_cutoff
        val marker_gene_method

    output:
        path 'dataset_deconvolved_marker_genes.h5ad', emit: result

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
    label 'small_mem'

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
        lambda
        n_marker_genes
        marker_gene_alpha_cutoff
        marker_gene_method

    main:
        DECONVOLVE(adata, n_components, lambda)

        SELECT_MARKER_GENES (DECONVOLVE.out.adata_output, DECONVOLVE.out.samples, n_marker_genes, marker_gene_alpha_cutoff, marker_gene_method)

        PLOT_DECONVOLUTION (SELECT_MARKER_GENES.out.result)

    emit:
        plots = PLOT_DECONVOLUTION.out.deconvolution_plots
        adata = SELECT_MARKER_GENES.out.result
        samples = DECONVOLVE.out.samples
}
