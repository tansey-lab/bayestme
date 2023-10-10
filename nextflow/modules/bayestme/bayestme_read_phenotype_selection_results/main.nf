process BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS {
    tag "$meta.id"
    label "process_single"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
    tuple val(meta), file(fold_results)

    output:
    tuple val(meta), env(LAMBDA), emit: lambda
    tuple val(meta), env(N_COMPONENTS), emit: n_components
    tuple val(meta), path("${prefix}/*.pdf"), emit: plots
    path  "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    mkdir "${prefix}"
    process_phenotype_selection_results \
        --plot-output "${prefix}" \
        --phenotype-selection-outputs ${fold_results} \
        --output-lambda lambda \
        --output-n-components n_components \
        ${args}

    LAMBDA=`cat lambda`
    N_COMPONENTS=`cat n_components`

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
