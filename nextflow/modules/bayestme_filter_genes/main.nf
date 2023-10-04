process BAYESTME_FILTER_GENES {
    tag "$meta.id"
    label 'process_single'
    label 'process_low'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"


    input:
    tuple val(meta) , path(adata)
    val filter_ribosomal_genes
    val n_top_by_standard_deviation
    val spot_threshold


    output:
    tuple val(meta), path("dataset_filtered.h5ad"), emit: adata_filtered
    path  "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ""
    def filter_ribosomal_genes_flag = params.bayestme_filter_ribosomal_genes == null ? "" : "--filter-ribosomal-genes"
    def n_top_by_standard_deviation_flag = params.bayestme_n_top_by_standard_deviation == null ? "": "--n-top-by-standard-deviation ${params.bayestme_n_top_by_standard_deviation}"
    def spot_threshold_flag = params.bayestme_spot_threshold == null ? "" : "--spot-threshold ${params.bayestme_spot_threshold}"
    """
    filter_genes --adata ${dataset} \
        --output dataset_filtered.h5ad \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'import bayestme;print(bayestme.__version__)' )
    END_VERSIONS
    """
}
