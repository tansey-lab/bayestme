process BLEEDING_CORRECTION {
    label 'process_high_memory'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    publishDir "${params.bayestme_outdir}/${sample_name}"

    input:
        path dataset
        val sample_name

    output:
        path "${sample_name}/dataset_filtered_corrected.h5ad", emit: adata_output
        path "${sample_name}/bleed_correction_results.h5", emit: bleed_correction_output

    script:
    def n_top_flag = params.bayestme_bleed_correction_n_top_genes == null ? "" : "--n-top ${params.bayestme_bleed_correction_n_top_genes}"
    def bleed_correction_n_em_steps_flag = params.bayestme_bleed_correction_n_em_steps == null ? "" : "--max-steps ${params.bayestme_bleed_correction_n_em_steps}"
    def bleed_correction_local_weight_flag = params.bayestme_bleed_correction_local_weight == null ? "" : "--local-weight ${params.bayestme_bleed_correction_local_weight}"
    """
    bleeding_correction --adata ${dataset} \
        ${n_top_flag} \
        ${bleed_correction_n_em_steps_flag} \
        ${bleed_correction_local_weight_flag} \
        --adata-output dataset_filtered_corrected.h5ad \
        --bleed-out bleed_correction_results.h5
    """
}

process PLOT_BLEEDING_CORRECTION {
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    publishDir "${params.outdir}/${sample_name}/plots/bleeding_correction"

    input:
        path filtered_anndata
        path bleed_corrected_anndata
        path bleed_correction_results
        val sample_name

    output:
        path "${sample_name}/plots/bleeding_correction/*.pdf", emit: result

    script:
    """
    plot_bleeding_correction --raw-adata ${filtered_anndata} \
        --corrected-adata ${bleed_corrected_anndata} \
        --bleed-correction-results ${bleed_correction_results} \
        --output-dir .
    """
}
