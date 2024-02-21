def create_lambda_values_flag(min_lambda, max_lambda) {
    // Create list of lambda values from min to max on log scale
    def lambda_values = []
    def current_lambda = min_lambda
    while (current_lambda <= max_lambda) {
        lambda_values.add(current_lambda)
        current_lambda *= 10
    }

    var lambda_values_flag = ""
    for (lambda_value in lambda_values) {
        lambda_values_flag += "--spatial-smoothing-values ${lambda_value} "
    }

    return lambda_values_flag
}

process BAYESTME_PHENOTYPE_SELECTION {
    tag "$meta.id"
    label 'process_high_memory'
    label 'process_long'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:89a932361287e7eaa9552d70908239d95479937e' :
        'docker.io/jeffquinnmsk/bayestme:89a932361287e7eaa9552d70908239d95479937e' }"

    input:
    tuple val(meta), path(adata), val(job_index), val(min_n_cell_types), val(max_n_cell_types), val(min_lambda), val(max_lambda), val(n_folds)

    output:
    tuple val(meta), path("${prefix}/fold_*.h5ad"), emit: result
    path  "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def n_components_min_flag = "--n-components-min ${min_n_cell_types}"
    def n_components_max_flag = "--n-components-max ${max_n_cell_types}"
    def phenotype_selection_spatial_smoothing_values_flag = create_lambda_values_flag(min_lambda, max_lambda)
    def n_folds_flag = "--n-fold ${n_folds}"
    """
    mkdir "${prefix}"
    phenotype_selection \
        --adata ${adata} \
        --output-dir "${prefix}" \
        --job-index ${job_index} \
        ${n_components_min_flag} \
        ${n_components_max_flag} \
        ${n_folds_flag} \
        ${phenotype_selection_spatial_smoothing_values_flag} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
