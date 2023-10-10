#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BAYESTME_PHENOTYPE_SELECTION } from '../../../../modules/bayestme/bayestme_phenotype_selection/main'

workflow bayestme_phenotype_selection {
    input = [ [ id:'test', single_end:false ], // meta map
              file(params.test_data['bayestme']['filtered_dataset'], checkIfExists: true),
              0,
              2,
              3,
              1,
              1000,
              2
            ]
    BAYESTME_PHENOTYPE_SELECTION ( input )
}
