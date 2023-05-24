nextflow.enable.dsl=2

process load_spaceranger {
    memory '8 GB'
    time '1h'


    input:
        path spaceranger_output_dir

    output:
        path 'dataset.h5ad', emit: result

    script:
    """
    load_spaceranger --input ${spaceranger_input_dir} --output dataset.h5ad
    """
}

def create_expression_truth_flag(expression_truth_values) {
    if (expression_truth_values == null || expression_truth_values.length == 0) {
        return ""
    } else {
        var expression_truth_flag = ""
        for (expression_truth_value in expression_truth_values) {
            expression_truth_flag += "--expression-truth ${expression_truth_value} "
        }

        return expression_truth_flag
    }
}

process filter_genes {
    memory '8 GB'
    time '1h'


    input:
        path dataset

    output:
        path 'dataset_filtered.h5ad', emit: result

    script:
    def filter_ribosomal_genes_flag = params.filter_ribosomal_genes == null ? "" : "--filter-ribosomal-genes"
    def n_top_by_standard_deviation_flag = params.n_top_by_standard_deviation == null ? "": "--n-top-by-standard-deviation ${params.n_top_by_standard_deviation}"
    def spot_threshold_flag = params.spot_threshold == null ? "" : "--spot-threshold ${params.spot_threshold}"
    def expression_truth_flag = create_expression_truth_flag(params.expression_truth_files)
    """
    filter_genes --adata ${dataset} \
        ${filter_ribosomal_genes_flag} \
        ${n_top_by_standard_deviation_flag} \
        ${spot_threshold_flag} \
        ${expression_truth_flag} \
        --output dataset_filtered.h5ad
    """
}

process bleeding_correction {
    memory '64 GB'
    time '24h'


    input:
        path dataset

    output:
        path 'dataset_filtered_corrected.h5ad', emit: adata_output
        path 'bleed_correction_results.h5', emit: bleed_correction_output

    script:
    def n_top_flag = params.bleed_correction_n_top_genes == null ? "" : "--n-top ${params.bleed_correction_n_top_genes}"
    def bleed_correction_n_em_steps_flag = params.bleed_correction_n_em_steps == null ? "" : "--max-steps ${params.bleed_correction_n_em_steps}"
    def bleed_correction_local_weight_flag = params.bleed_correction_local_weight == null ? "" : "--local-weight ${params.bleed_correction_local_weight}"
    """
    bleeding_correction --adata ${dataset} \
        ${n_top_flag} \
        ${bleed_correction_n_em_steps_flag} \
        ${bleed_correction_local_weight_flag} \
        --adata-output dataset_filtered_corrected.h5ad \
        --bleed-out bleed_correction_results.h5
    """
}

process plot_bleeding_correction {
    memory '8 GB'
    time '1h'


    input:
        path filtered_anndata
        path bleed_corrected_anndata
        path bleed_correction_results

    output:
        path '*.pdf', emit: result

    script:
    """
    plot_bleeding_correction --raw-adata ${filtered_anndata} \
        --corrected-adata ${bleed_corrected_anndata} \
        --bleed-correction-results ${bleed_correction_results} \
        --output-dir .
    """
}

def create_lambda_values_flag(lambda_values) {
    if (lambda_values == null) {
        return ""
    } else {
        var lambda_values_flag = ""
        for (lambda_value in lambda_values) {
            lambda_values_flag += "--lambda-values ${lambda_value} "
        }

        return lambda_values_flag
    }
}

process phenotype_selection {
    memory '96 GB'
    time '96h'


    input:
        val job_index
        path adata

    output:
        path 'fold_*.h5ad', emit: result

    script:
    def n_fold_flag = "--n-fold ${params.phenotype_selection_n_fold}"
    def n_splits_flag = "--n-splits ${params.phenotype_selection_n_splits}"
    def n_samples_flag = "--n-samples ${params.phenotype_selection_n_samples}"
    def n_burn_flag = "--n-burn ${params.phenotype_selection_n_burn}"
    def n_thin_flag = "--n-thin ${params.phenotype_selection_n_thin}"
    def n_gene_flag = "--n-gene ${params.phenotype_selection_n_gene}"
    def n_components_min_flag = "--n-components-min ${params.phenotype_selection_n_components_min}"
    def phenotype_selection_lambda_values_flag = create_lambda_values_flag(params.phenotype_selection_lambda_values)
    def phenotype_selection_max_ncell_flag = "--max-ncell ${params.phenotype_selection_max_ncell}"
    def phenotype_selection_background_noise_flag = params.background_noise ? "--background-noise" : ""
    def phenotype_selection_lda_initialization_flag = params.lda_initialization ? "--lda-initialization" : ""
    """
    phenotype_selection --adata ${adata} \
        --output-dir . \
        --job-index ${job_index} \
        ${n_fold_flag} \
        ${n_splits_flag} \
        ${n_samples_flag} \
        ${n_burn_flag} \
        ${n_thin_flag} \
        ${n_gene_flag} \
        ${n_components_min_flag} \
        ${phenotype_selection_lambda_values_flag} \
        ${phenotype_selection_max_ncell_flag} \
        ${phenotype_selection_background_noise_flag} \
        ${phenotype_selection_lda_initialization_flag}
    """
}

process deconvolve {
    memory '96 GB'
    time '96h'


    input:
        path adata
        val n_components
        val lambda

    output:
        path 'dataset_deconvolved.h5ad', emit: adata_output
        path 'deconvolution_samples.h5', emit: samples

    script:
    def n_samples_flag = "--n-samples ${params.deconvolution_n_samples}"
    def n_burn_flag = "--n-burn ${params.deconvolution_n_burn}"
    def n_thin_flag = "--n-thin ${params.deconvolution_n_thin}"
    def n_gene_flag = "--n-gene ${params.deconvolution_n_gene}"
    def phenotype_selection_background_noise_flag = params.phenotype_selection_background_noise_flag ? "--background-noise" : ""
    def phenotype_selection_lda_initialization_flag = params.phenotype_selection_lda_initialization ? "--lda-initialization" : ""
    def expression_truth_flag = create_expression_truth_flag(params.expression_truth_files)
    """
    deconvolve --adata dataset_filtered_corrected.h5ad \
        --adata-output dataset_deconvolved.h5ad \
        --output deconvolution_samples.h5 \
        --lam2 ${lambda} \
        --n-components ${n_components} \
        ${n_samples_flag} \
        ${n_burn_flag} \
        ${n_thin_flag} \
        ${n_gene_flag} \
        ${phenotype_selection_background_noise_flag} \
        ${phenotype_selection_lda_initialization_flag} \
        ${expression_truth_flag}
    """
}

process select_marker_genes {
    memory '64 GB'
    time '2h'


    input:
        path adata
        path deconvolution_samples

    output:
        path 'dataset_deconvolved_marker_genes.h5ad', emit: result

    script:
    """
    select_marker_genes --adata ${adata} \
        --adata-output dataset_deconvolved_marker_genes.h5ad \
        --deconvolution-result ${deconvolution_samples} \
        --n-marker-genes ${params.n_marker_genes} \
        --alpha ${params.marker_gene_alpha_cutoff} \
        --marker-gene-method ${params.marker_gene_method}
    """
}

process plot_deconvolution {
    memory '64 GB'
    time '1h'


    input:
        path adata

    output:
        path '*.pdf', emit: deconvolution_plots

    script:
    """
    plot_deconvolution --adata ${adata} \
        --output-dir .
    """
}

process spatial_expression {
    memory '96 GB'
    time '96h'


    input:
        path adata
        path deconvolution_samples

    output:
        path 'sde_samples.h5', emit: samples

    script:
    def n_spatial_patterns_flag = "--n-spatial-patterns ${params.spatial_expression_n_spatial_patterns}"
    def n_samples_flag = "--n-samples ${params.deconvolution_n_samples}"
    def n_burn_flag = "--n-burn ${params.deconvolution_n_burn}"
    def n_thin_flag = "--n-thin ${params.deconvolution_n_thin}"
    def n_gene_flag = "--n-gene ${params.deconvolution_n_gene}"
    def simple_flag = params.use_simple_spatial_expression_model ? "--simple" : ""
    def alpha0_flag = "--alpha0 ${params.spatial_expression_alpha0}"
    def prior_var_flag = "--prior-var ${params.spatial_expression_prior_var}"
    def n_cell_min_flag = "--n-cell-min ${params.spatial_expression_n_cell_min}"
    """
    spatial_expression --adata ${adata} \
        --output sde_samples.h5 \
        --deconvolve-results ${deconvolution_samples} \
        ${n_spatial_patterns_flag} \
        ${n_samples_flag} \
        ${n_burn_flag} \
        ${n_thin_flag} \
        ${n_gene_flag} \
        ${simple_flag} \
        ${alpha0_flag} \
        ${prior_var_flag} \
        ${n_cell_min_flag}
    """
}

def create_cell_type_names_flag(cell_type_names) {
    if (cell_type_names == null || cell_type_names.length == 0) {
        return ""
    } else {
        var cell_type_names_flag = "--cell-type-names "
        for (cell_type_name in cell_type_names) {
            cell_type_names_flag += "${cell_type_name},"
        }
        return cell_type_names_flag[0..-2]
    }
}

process plot_spatial_expression {
    memory '64 GB'
    time '1h'


    input:
        path sde_samples
        path deconvolution_samples
        path adata

    output:
        path '*.pdf', emit: result, optional: true

    script:
    def cell_type_names_flag = create_cell_type_names_flag(params.cell_type_names)
    """
    plot_spatial_expression --adata ${adata} \
        --deconvolution-result ${deconvolution_samples} \
        --sde-result ${sde_samples} \
        ${cell_type_names_flag} \
        --output-dir .
    """
}

process read_phenotype_selection_results {
    memory '64 GB'
    time '1h'


    input:
        path phenotype_selection_result
    output:
        env LAMBDA, emit: lambda
        env N_COMPONENTS, emit: n_components

    script:
    def lambda_values_flag = create_lambda_values_flag(params.phenotype_selection_lambda_values)
    """
    read_phenotype_selection_results \
        --phenotype-selection-outputs ${phenotype_selection_result}* \
        ${lambda_values_flag} \
        --output-lambda lambda \
        --output-n-components n_components

    LAMBDA=`cat lambda`
    N_COMPONENTS=`cat n_components`
    """
}

def calculate_n_phenotype_selection_jobs(lambdas, min_n_components, max_n_components, n_folds) {
    log.info "${lambdas}"
    return lambdas.size() * (max_n_components - min_n_components) * n_folds
}

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

        phenotype_selection(job_indices, bleeding_correction.out.adata_output)

        read_phenotype_selection_results( phenotype_selection.out.result.collect() )
    } else {
        log.info "Got values ${params.lambda} and ${params.n_components} for lambda and n_components, will skip phenotype selection."
    }

    def n_components = params.n_components == null ? read_phenotype_selection_results.out.n_components : params.n_components
    def lambda = params.lambda == null ? read_phenotype_selection_results.out.lambda : params.lambda

    deconvolve(
        bleeding_correction.out.adata_output,
        n_components,
        lambda
    )

    select_marker_genes(deconvolve.out.adata_output, deconvolve.out.samples)

    plot_deconvolution(select_marker_genes.out.result )

    spatial_expression( select_marker_genes.out.result, deconvolve.out.samples )

    plot_spatial_expression( spatial_expression.out.samples, deconvolve.out.samples, select_marker_genes.out.result )
}
