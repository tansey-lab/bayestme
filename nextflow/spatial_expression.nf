process SPATIAL_EXPRESSION {
    label 'process_high_memory'
    label 'process_long'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.outdir}/${sample_name}"

    input:
        path adata
        path deconvolution_samples
        val sample_name

    output:
        path 'sde_samples.h5', emit: samples

    script:
    def n_spatial_patterns_flag = "--n-spatial-patterns ${params.bayestme_spatial_expression_n_spatial_patterns}"
    def n_samples_flag = "--n-samples ${params.bayestme_spatial_expression_n_samples}"
    def n_burn_flag = "--n-burn ${params.bayestme_spatial_expression_n_burn}"
    def n_thin_flag = "--n-thin ${params.bayestme_spatial_expression_n_thin}"
    def n_gene_flag = "--n-gene ${params.bayestme_spatial_expression_n_genes}"
    def simple_flag = params.bayestme_use_simple_spatial_expression_model ? "--simple" : ""
    def alpha0_flag = "--alpha0 ${params.bayestme_spatial_expression_alpha0}"
    def prior_var_flag = "--prior-var ${params.bayestme_spatial_expression_prior_var}"
    def n_cell_min_flag = "--n-cell-min ${params.bayestme_spatial_expression_n_cell_min}"
    def seed_flag = "--seed ${params.bayestme_seed}"
    """
    spatial_expression --adata ${adata} \
        --output sde_samples.h5 \
        --deconvolve-results ${deconvolution_samples} \
        ${n_spatial_patterns_flag} \
        ${n_samples_flag} \
        ${n_burn_flag} \
        ${n_thin_flag} \
        ${n_gene_flag} \
        ${simple_flag} \
        ${alpha0_flag} \
        ${prior_var_flag} \
        ${n_cell_min_flag}
    """
}


process PLOT_SPATIAL_EXPRESSION {
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    publishDir "${params.outdir}/${sample_name}/plots/spatial_expression"

    input:
        path sde_samples
        path deconvolution_samples
        path adata
        val sample_name

    output:
        path '*.pdf', emit: result, optional: true

    script:
    """
    plot_spatial_expression --adata ${adata} \
        --deconvolution-result ${deconvolution_samples} \
        --sde-result ${sde_samples} \
        --moran-i-score-threshold ${params.bayestme_significant_spatial_pattern_moran_i_score_threshold} \
        --tissue-threshold ${params.bayestme_significant_spatial_pattern_tissue_threshold} \
        --gene-spatial-pattern-proportion-threshold ${params.bayestme_significant_spatial_pattern_gene_spatial_pattern_proportion_threshold} \
        --output-dir .
    """
}
