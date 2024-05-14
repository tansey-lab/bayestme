include { BAYESTME_LOAD_SPACERANGER } from '../../../modules/bayestme/bayestme_load_spaceranger/main'
include { BAYESTME_FILTER_GENES } from '../../../modules/bayestme/bayestme_filter_genes/main'
include { BAYESTME_BLEEDING_CORRECTION } from '../../../modules/bayestme/bayestme_bleeding_correction/main'
include { BAYESTME_DECONVOLUTION } from '../../../modules/bayestme/bayestme_deconvolution/main'
include { BAYESTME_SPATIAL_TRANSCRIPTIONAL_PROGRAMS } from '../../../modules/bayestme/bayestme_spatial_transcriptional_programs/main'

workflow BAYESTME_BASIC_VISIUM_ANALYSIS {

    take:
    ch_input  // channel: [ val(meta), path(spaceranger_dir), val(n_cell_types) ]

    main:

    BAYESTME_LOAD_SPACERANGER( ch_input.map { tuple(it[0], it[1]) } )

    filter_genes_input = BAYESTME_LOAD_SPACERANGER.out.adata.map { tuple(
        it[0],
        it[1],
        true,
        1000,
        0.9,
        [])
    }

    BAYESTME_FILTER_GENES( filter_genes_input )

    def deconvolution_input = null

    if (params.bleeding_correction) {
        BAYESTME_BLEEDING_CORRECTION( BAYESTME_FILTER_GENES.out.adata_filtered )

        deconvolution_input = BAYESTME_BLEEDING_CORRECTION.out.adata_corrected
            .join( ch_input.map { tuple(it[0], it[2]) } )
            .map { tuple(it[0], it[1], it[2], 1.0, []) }
    } else {
        deconvolution_input = BAYESTME_FILTER_GENES.out.adata_filtered
            .join( ch_input.map { tuple(it[0], it[2]) } )
            .map { tuple(it[0], it[1], it[2], 1.0, []) }
    }

    BAYESTME_DECONVOLUTION( deconvolution_input )

    BAYESTME_DECONVOLUTION.out.adata_deconvolved.join(BAYESTME_DECONVOLUTION.out.deconvolution_samples)
        .map { tuple(it[0], it[1], it[2], []) }
        .tap { stp_input }

    BAYESTME_SPATIAL_TRANSCRIPTIONAL_PROGRAMS( stp_input )

    emit:
    adata                      = BAYESTME_DECONVOLUTION.out.adata_deconvolved
    marker_gene_lists          = BAYESTME_DECONVOLUTION.out.marker_gene_lists
    deconvolution_plots        = BAYESTME_DECONVOLUTION.out.plots
    versions                   = BAYESTME_DECONVOLUTION.out.versions
}
