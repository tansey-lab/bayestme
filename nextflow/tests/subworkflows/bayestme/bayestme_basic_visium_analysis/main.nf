#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { BAYESTME_BASIC_VISIUM_ANALYSIS } from '../../../../subworkflows/bayestme/bayestme_basic_visium_analysis/main'

workflow bayestme_basic_visium_analysis {
    input = Channel.fromList([tuple([ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['spaceranger'], checkIfExists: true),
              2
            )])
    BAYESTME_BASIC_VISIUM_ANALYSIS ( input )
}
