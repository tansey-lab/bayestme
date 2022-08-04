FROM satijalab/seurat:latest

RUN R -e "install.packages(c('optparse', 'remotes'), dependencies=TRUE)" && \
    R -e "remotes::install_github('haotian-zhuang/findPC')"

COPY /seurat_fine_mapping/entrypoint.sh /

COPY /seurat_fine_mapping/cluster_relative_expression.R /

ENTRYPOINT ["/entrypoint.sh"]