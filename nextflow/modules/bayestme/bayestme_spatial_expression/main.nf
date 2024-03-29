process BAYESTME_SPATIAL_EXPRESSION {
    tag "$meta.id"
    label 'process_high_memory'
    label 'process_long'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:23a8cad61219c103ff8384f79f5c53734e71ad81' :
        'docker.io/jeffquinnmsk/bayestme:23a8cad61219c103ff8384f79f5c53734e71ad81' }"

    input:
    tuple val(meta), path(adata), path(deconvolution_samples)

    output:
    tuple val(meta), path("${prefix}/sde_samples.h5")                  , emit: sde_samples
    tuple val(meta), path("${prefix}/plots/*")                         , emit: plots, optional: true
    path  "versions.yml"                                               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def args2 = task.ext.args2 ?: ""
    """
    mkdir -p "${prefix}/plots"

    spatial_expression --adata ${adata} \
        --output ${prefix}/sde_samples.h5 \
        --deconvolve-results ${deconvolution_samples} \
        ${args}

    plot_spatial_expression \
        --deconvolution-result ${deconvolution_samples} \
        --adata ${adata} \
        --sde-result ${prefix}/sde_samples.h5 \
        --output-dir ${prefix}/plots \
        ${args2}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
