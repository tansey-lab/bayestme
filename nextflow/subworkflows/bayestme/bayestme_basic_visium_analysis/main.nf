include { BAYESTME_LOAD_SPACERANGER } from '../../../modules/bayestme/bayestme_load_spaceranger/main'
include { BAYESTME_FILTER_GENES } from '../../../modules/bayestme/bayestme_filter_genes/main'
include { BAYESTME_BLEEDING_CORRECTION } from '../../../modules/bayestme/bayestme_bleeding_correction/main'
include { BAYESTME_DECONVOLUTION } from '../../../modules/bayestme/bayestme_deconvolution/main'


workflow BAYESTME_BASIC_VISIUM_ANALYSIS {

    take:
    ch_input  // channel: [ val(meta), path(spaceranger_dir), val(n_cell_types) ]

    main:

    BAYESTME_LOAD_SPACERANGER( ch_input.map { tuple(it[0], it[1]) } )

    filter_genes_input = BAYESTME_LOAD_SPACERANGER.out.adata.map { tuple(it[0], it[1], true, 1000, 0.9, []) }

    BAYESTME_FILTER_GENES( filter_genes_input )

    bleed_correction_input = BAYESTME_FILTER_GENES.out.adata_filtered.map { tuple(it[0], it[1], 1000) }

    BAYESTME_BLEEDING_CORRECTION( bleed_correction_input )

    deconvolution_input = BAYESTME_BLEEDING_CORRECTION.out.adata_corrected
        .join( ch_input.map { tuple(it[0], it[2]) } )
        .map { tuple(it[0], it[1], it[2], 1000.0, []) }

    BAYESTME_DECONVOLUTION( deconvolution_input )

    emit:
    adata                      = BAYESTME_DECONVOLUTION.out.adata_deconvolved
    marker_gene_lists          = BAYESTME_DECONVOLUTION.out.marker_gene_lists
    deconvolution_plots        = BAYESTME_DECONVOLUTION.out.plots
    versions                   = BAYESTME_DECONVOLUTION.out.versions
}
