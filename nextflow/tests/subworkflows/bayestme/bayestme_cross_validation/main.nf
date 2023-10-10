#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { BAYESTME_CROSS_VALIDATION } from '../../../../subworkflows/bayestme/bayestme_cross_validation/main'

workflow bayestme_cross_validation {
    input = Channel.fromList([tuple([ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['filtered_dataset'], checkIfExists: true),
              2,
              3,
              1,
              10,
              2
            )])
    BAYESTME_CROSS_VALIDATION ( input )
}
