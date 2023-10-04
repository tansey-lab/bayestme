process BAYESTME_FILTER_GENES {
    tag "$meta.id"
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
    tuple val(meta) , path(adata)
    val filter_ribosomal_genes
    val n_top_by_standard_deviation
    val spot_threshold
    path expression_truth // optional

    output:
    tuple val(meta), path("dataset_filtered.h5ad"), emit: adata_filtered
    path  "versions.yml"                , emit: versions


    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ""
    def filter_ribosomal_genes_flag = filter_ribosomal_genes ? "" : "--filter-ribosomal-genes"
    def n_top_by_standard_deviation_flag = "--n-top-by-standard-deviation ${n_top_by_standard_deviation}"
    def spot_threshold_flag = "--spot-threshold ${spot_threshold}"
    """
    filter_genes --adata ${adata} \
        ${filter_ribosomal_genes_flag} \
        ${n_top_by_standard_deviation_flag} \
        ${spot_threshold_flag} \
        --output dataset_filtered.h5ad \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'import bayestme;print(bayestme.__version__)' )
    END_VERSIONS
    """
}
