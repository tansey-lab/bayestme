process CELLTYPIST_ANNOTATION {

    tag "Running Celltypist annotation"
    label 'process_medium'

    container 'oandrefonseca/scratch-annotation:main'
    publishDir "${params.outdir}/${params.project_name}", mode: 'copy', overwrite: true

    input:
        path(notebook)
        path(anndata_object)
        path(config)

    output:
        path("_freeze/${notebook.baseName}")                                  , emit: cache
        path("data/${params.project_name}_celltypist_annotation_object.h5ad") , emit: ann_object
        path("data/Immune_All")                                               , emit: csv_file
        path("report/${notebook.baseName}.html")                              , emit: html
        path("_freeze/**/figure-html/*.png")                                  , emit: figures
    when:
        task.ext.when == null || task.ext.when

    script:
        def param_file = task.ext.args ? "-P anndata_object:${anndata_object} -P ${task.ext.args}" : ""
        """
        quarto render ${notebook} ${param_file}
        """
    stub:
        def param_file = task.ext.args ? "-P anndata_object:${anndata_object} -P ${task.ext.args}" : ""
        """
        mkdir -p data _freeze/${notebook.baseName}
        mkdir -p _freeze/DUMMY/figure-html

        touch _freeze/DUMMY/figure-html/FILE.png


        touch data/${params.project_name}_celltypist_annotation_object.h5ad
        mkdir -p _freeze/${notebook.baseName} data/Immune_All
        touch _freeze/${notebook.baseName}/${notebook.baseName}.html

        mkdir -p report
        touch report/${notebook.baseName}.html

        echo ${param_file} > _freeze/${notebook.baseName}/params.yml
        """

}
