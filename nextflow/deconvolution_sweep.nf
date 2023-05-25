include { DECONVOLUTION } from './deconvolution'

workflow {
    // Run deconvolution and plots for sweep of cell type numbers
    def n_component_numbers = Channel.of(2..11)
    var adata = file( params.input_adata )

    DECONVOLUTION (
        adata,
        n_component_numbers,
        params.lambda,
        params.n_marker_genes,
        params.marker_gene_alpha_cutoff,
        params.marker_gene_method
    )
}
