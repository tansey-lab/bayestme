#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYESTME_SPATIAL_EXPRESSION } from '../../../../modules/bayestme/bayestme_spatial_expression/main'

workflow bayestme_spatial_expression {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['deconvolved_dataset'], checkIfExists: true),
              file(params.test_data['bayestme']['deconvolution_samples'], checkIfExists: true),
            ]
    BAYESTME_SPATIAL_EXPRESSION ( input )
}
