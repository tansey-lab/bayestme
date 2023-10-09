#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS } from '../../../../modules/bayestme/bayestme_read_phenotype_selection_results/main'

workflow bayestme_read_phenotype_selection_results {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['phenotype_selection'], checkIfExists: true)
            ]
    BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS ( input )
}
