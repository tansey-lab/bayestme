process BAYESTME_LOAD_SPACERANGER {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:0ba49e50ffadee1007bb1aaec34cf50a39b245df' :
        'docker.io/jeffquinnmsk/bayestme:0ba49e50ffadee1007bb1aaec34cf50a39b245df' }"

    input:
    tuple val(meta), path(spaceranger_dir)

    output:
    tuple val(meta), path("${prefix}/dataset.h5ad"), emit: adata
    path  "versions.yml"                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir "${prefix}"
    load_spaceranger --input ${spaceranger_dir} \
        --output "${prefix}/dataset.h5ad" \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
