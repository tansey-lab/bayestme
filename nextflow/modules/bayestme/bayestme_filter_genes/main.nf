process BAYESTME_FILTER_GENES {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
    tuple val(meta), path(adata), val(filter_ribosomal_genes), val(n_top_by_standard_deviation), val(spot_threshold), path(expression_truth)

    output:
    tuple val(meta), path("${prefix}/dataset_filtered.h5ad"), emit: adata_filtered
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def filter_ribosomal_genes_flag = filter_ribosomal_genes ? "" : "--filter-ribosomal-genes"
    def n_top_by_standard_deviation_flag = "--n-top-by-standard-deviation ${n_top_by_standard_deviation}"
    def spot_threshold_flag = "--spot-threshold ${spot_threshold}"
    def expression_truth_flag = expression_truth ? "--expression-truth ${task.ext.expression_truth}" : ""
    """
    mkdir "${prefix}"
    filter_genes --adata ${adata} \
        ${filter_ribosomal_genes_flag} \
        ${n_top_by_standard_deviation_flag} \
        ${spot_threshold_flag} \
        ${expression_truth_flag} \
        --output "${prefix}/dataset_filtered.h5ad" \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
