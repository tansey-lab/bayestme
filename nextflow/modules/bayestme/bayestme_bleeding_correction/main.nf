process BAYESTME_BLEEDING_CORRECTION {
    tag "$meta.id"
    label 'process_high_memory'
    label 'process_long'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest' :
        'docker.io/jeffquinnmsk/bayestme:latest' }"

    input:
    tuple val(meta), path(adata), val(n_top_genes)

    output:
    tuple val(meta), path("dataset_corrected.h5ad")     , emit: adata_corrected
    tuple val(meta), path("bleed_correction_results.h5"), emit: bleed_correction_output
    tuple val(meta), path("plots/*")                    , emit: plots
    path  "versions.yml"                                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ""
    def args2 = task.ext.args2 ?: ""
    def n_top_flag = "--n-top ${n_top_genes}"
    """
    bleeding_correction --adata ${adata} \
        ${n_top_flag} \
        --adata-output dataset_corrected.h5ad \
        --bleed-out bleed_correction_results.h5 \
        ${args}

    mkdir plots
    plot_bleeding_correction --raw-adata ${adata} \
        --corrected-adata dataset_corrected.h5ad \
        --bleed-correction-results bleed_correction_results.h5 \
        ${args2} \
        --output-dir ./plots

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
