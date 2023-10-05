#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYESTME_DECONVOLUTION } from '../../../../modules/bayestme/bayestme_deconvolution/main'

workflow bayestme_deconvolution {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['filtered_dataset'], checkIfExists: true),
              5,
              1.0
            ]
    BAYESTME_DECONVOLUTION ( input, [] )
}
