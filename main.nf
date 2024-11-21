include { BAYESTME_BASIC_VISIUM_ANALYSIS } from './nextflow/subworkflows/bayestme/bayestme_basic_visium_analysis/main'

workflow {
    BAYESTME_BASIC_VISIUM_ANALYSIS(
        Channel.fromList( [tuple([id: "sample", single_end: false],
        file(params.input),
        params.n_cell_types,
        params.reference_scrna ? [] : file(params.reference_scrna, checkIfExists: true)
        ) ])
    )
}
