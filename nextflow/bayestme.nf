include { DECONVOLUTION } from './deconvolution'
include { LOAD_INPUT } from './load_input'
include { FILTER_GENES } from './filter_genes'
include { BLEEDING_CORRECTION } from './bleeding_correction'
include { PLOT_BLEEDING_CORRECTION } from './bleeding_correction'
include { SPATIAL_EXPRESSION } from './spatial_expression'
include { PLOT_SPATIAL_EXPRESSION } from './spatial_expression'

workflow BAYESTME {
    take:
    input
    n_components

    main:
    LOAD_INPUT( input )

    FILTER_GENES( LOAD_INPUT.out.result, LOAD_INPUT.out.sample_name )

    BLEEDING_CORRECTION( FILTER_GENES.out.result, LOAD_INPUT.out.sample_name )

    PLOT_BLEEDING_CORRECTION( FILTER_GENES.out.result,
        BLEEDING_CORRECTION.out.adata_output,
        BLEEDING_CORRECTION.out.bleed_correction_output,
        LOAD_INPUT.out.sample_name )

    DECONVOLUTION( BLEEDING_CORRECTION.out.adata_output,
        LOAD_INPUT.out.sample_name,
        n_components
    )


    emit:
    adata = DECONVOLUTION.out.adata
    sample_name = LOAD_INPUT.out.sample_name
    deconvolution_samples = DECONVOLUTION.out.samples
    deconvolution_plots = DECONVOLUTION.out.plots
    bleeding_correction_results = BLEEDING_CORRECTION.out.bleed_correction_output
    bleeding_correction_plots = PLOT_BLEEDING_CORRECTION.out.result
    marker_gene_lists = DECONVOLUTION.out.marker_genes
}
