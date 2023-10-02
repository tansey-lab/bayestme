include { DECONVOLUTION } from './deconvolution'

process LOAD_INPUT {
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir params.outdir

    input:
        path input

    output:
        path '*.h5ad', emit: result
        val sample_name, emit: sample_name

    script:
        sample_name = input.getSimpleName()
        if (input.endsWith('.h5ad') && file(input).isFile())
            """
            cp ${input} ${input.getSimpleName()}.h5ad
            """
        else if (file(input).isDirectory())
            """
            load_spaceranger --input ${input} --output ${input.getSimpleName()}.h5ad
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

process FILTER_GENES {
    label 'process_single'
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.outdir}/${sample_name}"

    input:
        path dataset
        val sample_name

    output:
        path 'dataset_filtered.h5ad', emit: result

    script:
    def filter_ribosomal_genes_flag = params.bayestme_filter_ribosomal_genes == null ? "" : "--filter-ribosomal-genes"
    def n_top_by_standard_deviation_flag = params.bayestme_n_top_by_standard_deviation == null ? "": "--n-top-by-standard-deviation ${params.bayestme_n_top_by_standard_deviation}"
    def spot_threshold_flag = params.bayestme_spot_threshold == null ? "" : "--spot-threshold ${params.bayestme_spot_threshold}"
    def expression_truth_flag = create_expression_truth_flag(params.bayestme_expression_truth_files)
    """
    mkdir -p "${sample_name}"
    filter_genes --adata ${dataset} \
        ${filter_ribosomal_genes_flag} \
        ${n_top_by_standard_deviation_flag} \
        ${spot_threshold_flag} \
        ${expression_truth_flag} \
        --output "${sample_name}/dataset_filtered.h5ad"
    """
}

process BLEEDING_CORRECTION {
    label 'process_high_memory'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.bayestme_outdir}/${sample_name}"

    input:
        path dataset
        val sample_name

    output:
        path "${sample_name}/dataset_filtered_corrected.h5ad", emit: adata_output
        path "${sample_name}/bleed_correction_results.h5", emit: bleed_correction_output

    script:
    def n_top_flag = params.bayestme_bleed_correction_n_top_genes == null ? "" : "--n-top ${params.bayestme_bleed_correction_n_top_genes}"
    def bleed_correction_n_em_steps_flag = params.bayestme_bleed_correction_n_em_steps == null ? "" : "--max-steps ${params.bayestme_bleed_correction_n_em_steps}"
    def bleed_correction_local_weight_flag = params.bayestme_bleed_correction_local_weight == null ? "" : "--local-weight ${params.bayestme_bleed_correction_local_weight}"
    """
    bleeding_correction --adata ${dataset} \
        ${n_top_flag} \
        ${bleed_correction_n_em_steps_flag} \
        ${bleed_correction_local_weight_flag} \
        --adata-output dataset_filtered_corrected.h5ad \
        --bleed-out bleed_correction_results.h5
    """
}

process PLOT_BLEEDING_CORRECTION {
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.outdir}/${sample_name}/plots/bleeding_correction"

    input:
        path filtered_anndata
        path bleed_corrected_anndata
        path bleed_correction_results
        val sample_name

    output:
        path "${sample_name}/plots/bleeding_correction/*.pdf", emit: result

    script:
    """
    mkdir -p "${sample_name}/plots/bleeding_correction"
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
            lambda_values_flag += "--spatial-smoothing-values ${lambda_value} "
        }

        return lambda_values_flag
    }
}

process PHENOTYPE_SELECTION {
    label 'process_high_memory'
    label 'process_long'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.outdir}/${sample_name}/phenotype_selection"

    input:
        val job_index
        path adata
        val phenotype_selection_spatial_smoothing_values
        val n_components_min
        val n_components_max
        val sample_name

    output:
        path 'fold_*.h5ad', emit: result

    script:
    def n_fold_flag = "--n-fold ${params.bayestme_phenotype_selection_n_fold}"
    def n_splits_flag = "--n-splits ${params.bayestme_phenotype_selection_n_splits}"
    def n_samples_flag = "--n-samples ${params.bayestme_phenotype_selection_n_samples}"
    def n_burn_flag = "--n-burn ${params.bayestme_phenotype_selection_n_burn}"
    def n_thin_flag = "--n-thin ${params.bayestme_phenotype_selection_n_thin}"
    def n_components_min_flag = "--n-components-min ${n_components_min}"
    def n_components_max_flag = "--n-components-max ${n_components_max}"
    def phenotype_selection_spatial_smoothing_values_flag = create_lambda_values_flag(phenotype_selection_spatial_smoothing_values)
    def phenotype_selection_background_noise_flag = params.bayestme_background_noise ? "--background-noise" : ""
    def phenotype_selection_lda_initialization_flag = params.bayestme_lda_initialization ? "--lda-initialization" : ""
    def inference_type_flag = "--inference-type ${params.bayestme_inference_type}"
    """
    phenotype_selection --adata ${adata} \
        --output-dir . \
        --job-index ${job_index} \
        ${n_fold_flag} \
        ${n_splits_flag} \
        ${n_samples_flag} \
        ${n_burn_flag} \
        ${n_thin_flag} \
        ${n_components_min_flag} \
        ${n_components_max_flag} \
        ${phenotype_selection_spatial_smoothing_values_flag} \
        ${phenotype_selection_background_noise_flag} \
        ${phenotype_selection_lda_initialization_flag} \
        ${inference_type_flag}
    """
}

process SPATIAL_EXPRESSION {
    label 'process_high_memory'
    label 'process_long'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.outdir}/${sample_name}"

    input:
        path adata
        path deconvolution_samples
        val sample_name

    output:
        path 'sde_samples.h5', emit: samples

    script:
    def n_spatial_patterns_flag = "--n-spatial-patterns ${params.bayestme_spatial_expression_n_spatial_patterns}"
    def n_samples_flag = "--n-samples ${params.bayestme_spatial_expression_n_samples}"
    def n_burn_flag = "--n-burn ${params.bayestme_spatial_expression_n_burn}"
    def n_thin_flag = "--n-thin ${params.bayestme_spatial_expression_n_thin}"
    def n_gene_flag = "--n-gene ${params.bayestme_spatial_expression_n_genes}"
    def simple_flag = params.bayestme_use_simple_spatial_expression_model ? "--simple" : ""
    def alpha0_flag = "--alpha0 ${params.bayestme_spatial_expression_alpha0}"
    def prior_var_flag = "--prior-var ${params.bayestme_spatial_expression_prior_var}"
    def n_cell_min_flag = "--n-cell-min ${params.bayestme_spatial_expression_n_cell_min}"
    def seed_flag = "--seed ${params.bayestme_seed}"
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

process PLOT_SPATIAL_EXPRESSION {
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.outdir}/${sample_name}/plots/spatial_expression"

    input:
        path sde_samples
        path deconvolution_samples
        path adata
        val sample_name

    output:
        path '*.pdf', emit: result, optional: true

    script:
    def cell_type_names_flag = create_cell_type_names_flag(params.bayestme_cell_type_names)
    """
    plot_spatial_expression --adata ${adata} \
        --deconvolution-result ${deconvolution_samples} \
        --sde-result ${sde_samples} \
        ${cell_type_names_flag} \
        --moran-i-score-threshold ${params.bayestme_significant_spatial_pattern_moran_i_score_threshold} \
        --tissue-threshold ${params.bayestme_significant_spatial_pattern_tissue_threshold} \
        --gene-spatial-pattern-proportion-threshold ${params.bayestme_significant_spatial_pattern_gene_spatial_pattern_proportion_threshold} \
        --output-dir .
    """
}

process READ_PHENOTYPE_SELECTION_RESULTS {
    label 'process_single'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://jeffquinnmsk/bayestme:latest':
        'docker.io/jeffquinnmsk/bayestme:latest' }"
    publishDir "${params.outdir}/${sample_name}/plots/phenotype_selection"

    input:
        path phenotype_selection_result
        val phenotype_selection_spatial_smoothing_values
        val sample_name
    output:
        env LAMBDA, emit: lambda
        env N_COMPONENTS, emit: n_components
        path "*.pdf", emit: plots

    script:
    def phenotype_selection_spatial_smoothing_values_flag = create_lambda_values_flag(phenotype_selection_spatial_smoothing_values)
    """
    process_phenotype_selection_results \
        --plot-output . \
        --phenotype-selection-outputs ${phenotype_selection_result}* \
        ${phenotype_selection_spatial_smoothing_values_flag} \
        --output-lambda lambda \
        --output-n-components n_components

    LAMBDA=`cat lambda`
    N_COMPONENTS=`cat n_components`
    """
}

def calculate_n_phenotype_selection_jobs(lambdas, min_n_components, max_n_components, n_folds) {
    log.info "${lambdas}"
    return lambdas.size() * ((max_n_components + 1) - min_n_components) * n_folds
}

workflow BAYESTME {
    take:
        input
        n_components

    main:
        LOAD_INPUT( input )

        FILTER_GENES( LOAD_INPUT.out.result, LOAD_INPUT.out.sample_name )

        BLEEDING_CORRECTION( FILTER_GENES.out.result, LOAD_INPUT.out.sample_name )

        PLOT_BLEEDING_CORRECTION(FILTER_GENES.out.result,
            BLEEDING_CORRECTION.out.adata_output,
            BLEEDING_CORRECTION.out.bleed_correction_output,
            LOAD_INPUT.out.sample_name )

        DECONVOLUTION (
            BLEEDING_CORRECTION.out.adata_output,
            LOAD_INPUT.out.sample_name,
            n_components,
            params.bayestme_spatial_smoothing_parameter,
            params.bayestme_n_marker_genes,
            params.bayestme_marker_gene_alpha_cutoff,
            params.bayestme_marker_gene_method,
            params.bayestme_deconvolution_use_spatial_guide,
            params.bayestme_inference_type
        )

        if (params.bayestme_run_spatial_expression) {
            SPATIAL_EXPRESSION( DECONVOLUTION.out.adata, DECONVOLUTION.out.samples, LOAD_INPUT.out.sample_name )
            PLOT_SPATIAL_EXPRESSION( SPATIAL_EXPRESSION.out.samples, DECONVOLUTION.out.samples, DECONVOLUTION.out.adata, LOAD_INPUT.out.sample_name )
        }
    emit:
        adata = DECONVOLUTION.out.adata
        sample_name = LOAD_INPUT.out.sample_name
        deconvolution_samples = DECONVOLUTION.out.samples
        deconvolution_plots = PLOT_DECONVOLUTION.out.deconvolution_plots
        bleeding_correction_results = BLEEDING_CORRECTION.out.bleed_correction_output
        bleeding_correction_plots = PLOT_BLEEDING_CORRECTION.out.result
        marker_gene_lists = SELECT_MARKER_GENES.out.csvs
        sde_samples = SPATIAL_EXPRESSION.out.samples, optional: true
        sde_plots = PLOT_SPATIAL_EXPRESSION.out.result, optional: true

}
