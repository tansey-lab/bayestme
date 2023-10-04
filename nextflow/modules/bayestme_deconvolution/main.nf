process BAYESTME_DECONVOLUTION {
    tag "$meta.id"
    label 'process_high_memory'
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
    tuple val(meta), path(adata)
    val n_cell_types
    val spatial_smoothing_parameter

    output:
    tuple val(meta), path("dataset_deconvolved.h5ad")     , emit: adata_deconvolved
    tuple val(meta), path("deconvolution_samples.h5")     , emit: deconvolution_samples
    tuple val(meta), path("plots", type: "dir")           , emit: plots
    path  "versions.yml"                                  , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ""
    def args2 = task.ext.args2 ?: ""
    def n_components_flag = "--n-components ${n_cell_types}"
    def spatial_smoothing_parameter_flag = "--spatial-smoothing-parameter ${spatial_smoothing_parameter}"
    def expression_truth_flag = task.ext.expression_truth == null ? "" : "--expression-truth ${task.ext.expression_truth}"
    """
    deconvolve --adata ${adata} \
        --adata-output dataset_deconvolved.h5ad \
        --output deconvolution_samples.h5 \
        ${spatial_smoothing_parameter_flag} \
        ${n_components_flag} \
        ${args}

    mkdir plots
    plot_deconvolution --adata dataset_deconvolved.h5ad \
        --output-dir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'import bayestme;print(bayestme.__version__)' )
    END_VERSIONS
    """
}
