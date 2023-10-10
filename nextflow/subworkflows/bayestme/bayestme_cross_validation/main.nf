include { BAYESTME_PHENOTYPE_SELECTION } from '../../../modules/bayestme/bayestme_phenotype_selection/main'
include { BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS } from '../../../modules/bayestme/bayestme_read_phenotype_selection_results/main'

def calculate_n_phenotype_selection_jobs(lambdas, min_n_components, max_n_components, n_folds) {
    log.info "${lambdas}"
    return lambdas.size() * ((max_n_components + 1) - min_n_components) * n_folds
}

def create_lambda_values(min_lambda, max_lambda) {
    log.info "min_lambda: ${min_lambda}"
    log.info "max_lambda: ${max_lambda}"
    // Create list of lambda values from min to max on log scale
    def lambda_values = []
    def lambda_value = min_lambda
    while (lambda_value <= max_lambda) {
        lambda_values.add(lambda_value)
        lambda_value = lambda_value * 10
    }

    return lambda_values
}

def create_job_index_sequence(
        min_n_cell_types,
        max_n_cell_types,
        min_lambda,
        max_lambda,
        n_folds) {
    lambdas = create_lambda_values(min_lambda, max_lambda)
    n_jobs = calculate_n_phenotype_selection_jobs(lambdas, min_n_cell_types, max_n_cell_types, n_folds)

    return (0..(n_jobs-1))
}

def create_tuples(meta,
        adata,
        min_n_cell_types,
        max_n_cell_types,
        min_lambda,
        max_lambda,
        n_folds) {
    job_indices = create_job_index_sequence(
            min_n_cell_types,
            max_n_cell_types,
            min_lambda,
            max_lambda,
            n_folds)

    output = []
    // iterate through job_indices
    for (job_index in job_indices) {
        output.add(
            tuple(meta,
                  adata,
                  job_index,
                  min_n_cell_types,
                  max_n_cell_types,
                  min_lambda,
                  max_lambda,
                  n_folds)
        )
    }

    return output
}

def get_job_size(meta,
        adata,
        min_n_cell_types,
        max_n_cell_types,
        min_lambda,
        max_lambda,
        n_folds) {

    job_indices = create_job_index_sequence(
            min_n_cell_types,
            max_n_cell_types,
            min_lambda,
            max_lambda,
            n_folds)

    return job_indices.size()
}

workflow BAYESTME_CROSS_VALIDATION {

    take:
    ch_adata  // channel: [ val(meta), path(adata), val(min_n_cell_types), val(max_n_cell_types), val(min_lambda), val(max_lambda), val(n_folds) ]

    main:
    ch_job_data = ch_adata.flatMap { create_tuples(it[0], it[1], it[2], it[3], it[4], it[5], it[6]) }
    job_sizes = ch_adata.map { tuple(it[0].id, get_job_size(it[0], it[1], it[2], it[3], it[4], it[5], it[6])) }

    ch_versions = Channel.empty()

    BAYESTME_PHENOTYPE_SELECTION(
        ch_job_data
    )

    grouped_results = BAYESTME_PHENOTYPE_SELECTION.out.result
        .groupTuple()
        .map { k, v -> tuple( groupKey(k, v.size()), v ) }

    BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS( grouped_results )

    emit:
    cv_lambda       = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.lambda
    cv_n_cell_types = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.n_components
    cv_plots        = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.plots
    versions        = BAYESTME_READ_PHENOTYPE_SELECTION_RESULTS.out.versions
}
