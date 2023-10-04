process BAYESTME_SPATIAL_EXPRESSION {
    label 'process_high_memory'
    label 'process_long'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
    tuple val(meta), path(adata)
    tuple val(meta), path(deconvolution_samples)

    output:
    tuple val(meta), path("sde_samples.h5")                            , emit: sde_samples
    tuple val(meta), path("plots/*")                                   , emit: plots
    path  "versions.yml"                                               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ""
    def args2 = task.ext.args2 ?: ""
    """
    spatial_expression --adata ${adata} \
        --output sde_samples.h5 \
        --deconvolve-results ${deconvolution_samples} \
        ${args}

    mkdir plots

    plot_spatial_expression \
        --deconvolution-result ${deconvolution_samples} \
        --sde-samples sde_samples.h5 \
        --output-dir ./plots \
        ${args2}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'import bayestme;print(bayestme.__version__)' )
    END_VERSIONS
    """
}
