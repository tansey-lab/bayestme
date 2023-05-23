
workflow {
    if (params.input_adata == null) {
        load_spaceranger(file(params.spaceranger_dir, type: dir))
    }

    var adata = params.input_adata == null ? load_spaceranger.out.result : file(params.input_adata)

    filter_genes(adata)

    bleeding_correction(filter_genes.out.result)

    plot_bleeding_correction(filter_genes.out.result,
        bleeding_correction.out.adata_output,
        bleeding_correction.out.bleed_correction_output)

    if (params.lambda == null && params.n_components == null) {
        log.info "No values supplied for lambda and n_components, will run phenotype selection."
        log.info "${params.phenotype_selection_lambda_values}"
        var n_phenotype_jobs = calculate_n_phenotype_selection_jobs(
            params.phenotype_selection_lambda_values,
            params.phenotype_selection_n_components_min,
            params.phenotype_selection_n_components_max,
            params.phenotype_selection_n_fold)

        log.info "Will need to run ${n_phenotype_jobs} jobs for phenotype selection."

        job_indices = Channel.of(0..(n_phenotype_jobs-1))

        phenotype_selection(job_indices, bleeding_correction.out.result)

        read_phenotype_selection_results( phenotype_selection.out.result.collect() )
    } else {
        log.info "Got values ${params.lambda} and ${params.n_components} for lambda and n_components, will skip phenotype selection."
    }

    def n_components = params.n_components == null ? read_phenotype_selection_results.out.n_components : params.n_components
    def lambda = params.lambda == null ? read_phenotype_selection_results.out.lambda : params.lambda

    deconvolve(
        bleeding_correction.out.adata_output,
        n_components
        lambda,
    )

    select_marker_genes(deconvolve.out.adata_output, deconvolve.out.samples)

    plot_deconvolution(select_marker_genes.out.adata_output)

    spatial_expression( select_marker_genes.out.result )

    plot_spatial_expression( spatial_expression.out.samples, deconvolve.out.samples, select_marker_genes.out.result )
}
