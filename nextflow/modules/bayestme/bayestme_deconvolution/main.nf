process BAYESTME_DECONVOLUTION {
    tag "$meta.id"
    label 'process_high_memory'
    label 'process_long'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:0ba49e50ffadee1007bb1aaec34cf50a39b245df' :
        'docker.io/jeffquinnmsk/bayestme:0ba49e50ffadee1007bb1aaec34cf50a39b245df' }"

    input:
    tuple val(meta), path(adata), val(n_cell_types), val(spatial_smoothing_parameter), path(expression_truth)

    output:
    tuple val(meta), path("${prefix}/dataset_deconvolved_marker_genes.h5ad")     , emit: adata_deconvolved
    tuple val(meta), path("${prefix}/deconvolution_samples.h5")                  , emit: deconvolution_samples
    tuple val(meta), path("${prefix}/plots/*")                                   , emit: plots
    tuple val(meta), path("${prefix}/*.csv")                                     , emit: marker_gene_lists
    path  "versions.yml"                                               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def args2 = task.ext.args2 ?: ""
    def args3 = task.ext.args3 ?: ""
    def n_components_flag = "--n-components ${n_cell_types}"
    def need_expression_truth_flag = !args.contains("--expression-truth")
    def spatial_smoothing_parameter_flag = "--spatial-smoothing-parameter ${spatial_smoothing_parameter}"
    def expression_truth_flag = expression_truth ? "--expression-truth ${expression_truth}" : ""

    expression_truth_flag = need_expression_truth_flag ? expression_truth_flag : ""
    """
    mkdir -p "${prefix}/plots"

    deconvolve --adata ${adata} \
        --adata-output "${prefix}/dataset_deconvolved.h5ad" \
        --output "${prefix}/deconvolution_samples.h5" \
        ${spatial_smoothing_parameter_flag} \
        ${n_components_flag} \
        ${expression_truth_flag} \
        ${args}

    select_marker_genes --adata "${prefix}/dataset_deconvolved.h5ad" \
        --adata-output "${prefix}/dataset_deconvolved_marker_genes.h5ad" \
        --deconvolution-result "${prefix}/deconvolution_samples.h5" \
        ${args2}

    mkdir plots
    plot_deconvolution --adata "${prefix}/dataset_deconvolved_marker_genes.h5ad" \
        --output-dir "${prefix}/plots" \
        ${expression_truth_flag} \
        ${args3}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
