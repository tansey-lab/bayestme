include { BAYESTME_PHENOTYPE_SELECTION } from '../../../modules/bayestme/bayestme_phenotype_selection/main'

def calculate_n_phenotype_selection_jobs(lambdas, min_n_components, max_n_components, n_folds) {
    log.info "${lambdas}"
    return lambdas.size() * ((max_n_components + 1) - min_n_components) * n_folds
}

def create_lambda_values(min_lambda, max_lambda) {
    // Create list of lambda values from min to max on log scale
    def lambda_values = []
    def lambda_value = min_lambda
    while (lambda_value <= max_lambda) {
        lambda_values.add(lambda_value)
        lambda_value *= 10
    }

    return lambda_values
}

def create_job_index_sequence(
        min_lambda,
        max_lambda,
        min_n_cell_types,
        max_n_cell_types,
        n_folds) {
    lambdas = create_lambda_values(min_lambda, max_lambda)
    n_jobs = calculate_n_phenotype_selection_jobs(lambdas, min_n_cell_types, max_n_cell_types, n_folds)

    return (0..(n_jobs-1))
}

def create_tuples(meta,
        adata,
        min_lambda,
        max_lambda,
        min_n_cell_types,
        max_n_cell_types,
        n_folds) {
    job_indices = create_job_index_sequence(
            min_lambda,
            max_lambda,
            min_n_cell_types,
            max_n_cell_types,
            n_folds)

    output = []
    // iterate through job_indices
    for (job_index in job_indices) {
        output.add(tuple(meta, adata, job_index))
    }

}

workflow BAYESTME_CROSS_VALIDATION {

    take:
    ch_adata        // channel: [ val(meta), path(adata),
                    //            val(min_n_cell_types), val(max_n_cell_types),
                    //            val(min_lambda), val(max_lambda), val(n_folds) ]

    main:
    // channel: [ val(meta), path(adata), val(job_index) ]
    ch_job_indices = ch_adata.flatMap { create_tuples(it[0], it[1], it[2], it[3], it[4], it[5], it[6])) }


    ch_versions = Channel.empty()

    lambdas = create_lambda_values_flag(min_lambda, max_lambda)

    BAYESTME_PHENOTYPE_SELECTION(
        ch_bam,
        ch_fasta,
        ch_fai
    )
    ch_versions = ch_versions.mix(BAYESTME_PHENOTYPE_SELECTION.out.versions.first())

    emit:
    aberrations_bed = WISECONDORX_PREDICT.out.aberrations_bed   // channel: [ val(meta), path(bed) ]
    bins_bed        = WISECONDORX_PREDICT.out.bins_bed          // channel: [ val(meta), path(bed) ]
    segments_bed    = WISECONDORX_PREDICT.out.segments_bed      // channel: [ val(meta), path(bed) ]
    chr_statistics  = WISECONDORX_PREDICT.out.chr_statistics    // channel: [ val(meta), path(txt) ]
    chr_plots       = WISECONDORX_PREDICT.out.chr_plots         // channel: [ val(meta), [ path(png), path(png), ... ] ]
    genome_plot     = WISECONDORX_PREDICT.out.genome_plot       // channel: [ val(meta), path(png) ]

    versions        = ch_versions                               // channel: path(versions.yml)
}
