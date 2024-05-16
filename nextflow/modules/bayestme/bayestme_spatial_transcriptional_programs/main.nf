process BAYESTME_SPATIAL_TRANSCRIPTIONAL_PROGRAMS {
    tag "$meta.id"
    label 'process_high_memory'
    label 'process_long'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        ('docker://jeffquinnmsk/bayestme:' + params.bayestme_version) :
        ('docker.io/jeffquinnmsk/bayestme:' + params.bayestme_version) }"

    input:
    tuple val(meta), path(adata), path(deconvolution_results), path(expression_truth)

    output:
    tuple val(meta), path("${prefix}/spatial_transcriptional_programs.h5")       , emit: stp
    tuple val(meta), path("${prefix}/stp_plots/*")                               , emit: plots
    path  "versions.yml"                                                         , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def expression_truth_flag = expression_truth ? "--expression-truth ${expression_truth}" : ""
    def need_expression_truth_flag = !args.contains("--expression-truth")
    expression_truth_flag = need_expression_truth_flag ? expression_truth_flag : ""
    """
    mkdir -p "${prefix}"

    spatial_transcriptional_programs --adata ${adata} \
        --deconvolution-result ${deconvolution_results} \
        --output "${prefix}/spatial_transcriptional_programs.h5" \
        ${expression_truth_flag} \
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bayestme: \$( python -c 'from importlib.metadata import version;print(version("bayestme"))' )
    END_VERSIONS
    """
}
