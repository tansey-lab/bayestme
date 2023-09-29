include { BAYESTME } from './nextflow/bayestme'

workflow {
    BAYESTME (params.spaceranger_dir)
}
