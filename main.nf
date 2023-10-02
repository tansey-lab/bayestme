include { BAYESTME, LOAD_SPACERANGER } from './nextflow/bayestme'

workflow {
    BAYESTME(
        params.input,
        params.n_components
    )
}
