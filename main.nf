include { BAYESTME } from './nextflow/bayestme'

workflow {
    BAYESTME(
        params.input,
        params.n_components
    )
}
