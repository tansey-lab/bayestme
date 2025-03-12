process BAYESTME_LOAD_SPACERANGER {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/bayestme:' + params.bayestme_version) :
        ('docker.io/jeffquinnmsk/bayestme:' + params.bayestme_version) }"

    input:
    tuple val(meta), path(spaceranger_dir)

    output:
    tuple val(meta), path("${prefix}/dataset.h5ad"), emit: adata
    tuple val(meta), path("${prefix}/*.pdf")       , emit: plots, optional: true
    path  "versions.yml"                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir "${prefix}"

    if [[ -f "${spaceranger_dir}" && "${spaceranger_dir}" == *.h5ad ]]; then
        echo "${spaceranger_dir} is already a h5ad file, creating a symlink"
        cp "${spaceranger_dir}" "${prefix}/dataset.h5ad"
    else
        load_spaceranger --input ${spaceranger_dir} \
            --output "${prefix}/dataset.h5ad" \
            ${args}
    fi

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
