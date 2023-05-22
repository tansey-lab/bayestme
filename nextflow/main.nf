nextflow.enable.dsl=2

process load_spaceranger {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        path spaceranger_output_dir

    output:
        path 'dataset.h5ad', emit: result

    script:
    """
    load_spaceranger --input ${spaceranger_input_dir} --output dataset.h5ad
    """
}

process filter_genes {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        path dataset

    output:
        path 'dataset_filtered.h5ad', emit: result

    script:
    """
    filter_genes --adata ${dataset} \
        --filter-ribosomal-genes \
        --n-top-by-standard-deviation 1000 \
        --output dataset_filtered.h5ad
    """
}

process bleeding_correction {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        path dataset

    output:
        path 'dataset_filtered_corrected.h5ad', emit: anndata_output
        path 'bleed_correction_results.h5', emit: bleed_correction_output

    script:
    """
    bleeding_correction --adata ${dataset} \
        --adata-output dataset_filtered_corrected.h5ad \
        --bleed-out bleed_correction_results.h5
    """
}

process plot_bleeding_correction {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        path filtered_anndata
        path bleed_corrected_anndata
        path bleed_correction_results

    output:
        path 'bleed_correction_results', type: dir, emit: result

    script:
    """
    plot_bleeding_correction --raw-adata ${filtered_anndata} \
        --corrected-adata ${bleed_corrected_anndata} \
        --bleed-correction-results ${bleed_correction_results} \
        --output-dir bleed_correction_results
    """
}

process phenotype_selection {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        val job_index
        path adata

    output:
        path 'fold_${job_index}.h5ad', emit: result

    script:
    """
    phenotype_selection --adata ${adata} \
        --output-dir . \
        --job-index ${job_index}
    """
}


process deconvolve {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        val lam2
        val n_components
        path adata

    output:
        path 'dataset_deconvolved.h5ad', emit: adata_output
        path 'deconvolution_samples.h5', emit: samples

    script:
    """
    deconvolve --adata dataset_filtered_corrected.h5ad \
        --adata-output dataset_deconvolved.h5ad \
        --output deconvolution_samples.h5 \
        --lam2 ${lam2} \
        --n-components ${n_components}
    """
}

process select_marker_genes {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        val n_marker_genes
        path adata
        path deconvolution_samples

    output:
        path 'dataset_deconvolved_marker_genes.h5ad', emit: adata_output

    script:
    """
    select_marker_genes --adata ${adata} \
        --adata-output dataset_deconvolved_marker_genes.h5ad \
        --deconvolution-result ${deconvolution_samples} \
        --n-marker-genes ${n_marker_genes}
    """
}

process plot_deconvolution {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        path adata

    output:
        path 'deconvolution_plots', type: dir, emit: adata_output

    script:
    """
    plot_deconvolution --adata ${adata} \
        --output-dir deconvolution_plots
    """
}

process spatial_expression {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        path adata

    output:
        path 'sde_samples.h5', emit: output

    script:
    """
    spatial_expression --adata ${adata} \
        --output sde_samples.h5
    """
}

process plot_spatial_expression {
    memory '8 GB'
    time '1h'
    container = "docker://jeffquinnmsk/bayestme:latest"

    input:
        path sde_samples
        path deconvolution_samples
        path adata

    output:
        path 'sde_plots', type: dir, emit: output

    script:
    """
    plot_spatial_expression --adata ${adata} \
        --deconvolution-result ${deconvolution_samples} \
        --sde-result ${sde_samples} \
        --output-dir sde_plots
    """
}
