process BAYESTME_BLEEDING_CORRECTION {
    tag "$meta.id"
    label 'process_high_memory'
    label 'process_long'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
    tuple val(meta), path(adata)

    output:
    tuple val(meta), path("${prefix}/dataset_corrected.h5ad")     , emit: adata_corrected
    tuple val(meta), path("${prefix}/bleed_correction_results.h5"), emit: bleed_correction_output
    tuple val(meta), path("${prefix}/plots/*")                    , emit: plots
    path  "versions.yml"                                          , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def args2 = task.ext.args2 ?: ""
    """
    mkdir -p "${prefix}/plots"

    bleeding_correction --adata ${adata} \
        --adata-output ${prefix}/dataset_corrected.h5ad \
        --bleed-out ${prefix}/bleed_correction_results.h5 \
        ${args}

    plot_bleeding_correction --raw-adata ${adata} \
        --corrected-adata ${prefix}/dataset_corrected.h5ad \
        --bleed-correction-results ${prefix}/bleed_correction_results.h5 \
        ${args2} \
        --output-dir ${prefix}/plots

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
