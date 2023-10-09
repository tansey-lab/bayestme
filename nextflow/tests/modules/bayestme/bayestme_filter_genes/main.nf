#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYESTME_FILTER_GENES } from '../../../../modules/bayestme/bayestme_filter_genes/main'

workflow bayestme_filter_genes {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['raw_dataset'], checkIfExists: true),
              false,
              3,
              0.5,
              []
            ]
    BAYESTME_FILTER_GENES ( input )
}
