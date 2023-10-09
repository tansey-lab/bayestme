#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYESTME_LOAD_SPACERANGER } from '../../../../modules/bayestme/bayestme_load_spaceranger/main'

workflow bayestme_load_spaceranger {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['spaceranger'], checkIfExists: true, type: 'dir')
            ]
    BAYESTME_LOAD_SPACERANGER ( input )
}
