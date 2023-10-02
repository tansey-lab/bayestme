process LOAD_INPUT {
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ? 'docker://jeffquinnmsk/bayestme:latest': 'docker.io/jeffquinnmsk/bayestme:latest' }"

    publishDir params.outdir

    input:
        path input

    output:
        path '*.h5ad', emit: result
        val sample_name, emit: sample_name

    script:
        sample_name = input.getSimpleName()
        if (input.endsWith('.h5ad') && file(input).isFile())
            """
            cp ${input} ${input.getSimpleName()}.h5ad
            """
        else if (input.isDirectory())
            """
            load_spaceranger --input ${input} --output ${input.getSimpleName()}.h5ad
            """
}
