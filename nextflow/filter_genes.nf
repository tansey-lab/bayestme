process FILTER_GENES {
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    publishDir "${params.outdir}/${sample_name}"

    input:
        path dataset
        val sample_name

    output:
        path 'dataset_filtered.h5ad', emit: result

    script:
    def filter_ribosomal_genes_flag = params.bayestme_filter_ribosomal_genes == null ? "" : "--filter-ribosomal-genes"
    def n_top_by_standard_deviation_flag = params.bayestme_n_top_by_standard_deviation == null ? "": "--n-top-by-standard-deviation ${params.bayestme_n_top_by_standard_deviation}"
    def spot_threshold_flag = params.bayestme_spot_threshold == null ? "" : "--spot-threshold ${params.bayestme_spot_threshold}"
    """
    mkdir -p "${sample_name}"
    filter_genes --adata ${dataset} \
        ${filter_ribosomal_genes_flag} \
        ${n_top_by_standard_deviation_flag} \
        ${spot_threshold_flag} \
        --output "${sample_name}/dataset_filtered.h5ad"
    """
}
