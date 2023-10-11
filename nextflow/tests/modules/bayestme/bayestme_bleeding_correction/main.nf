#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYESTME_BLEEDING_CORRECTION } from '../../../../modules/bayestme/bayestme_bleeding_correction/main'

workflow bayestme_bleeding_correction {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['filtered_dataset'], checkIfExists: true)
            ]
    BAYESTME_BLEEDING_CORRECTION ( input )
}
