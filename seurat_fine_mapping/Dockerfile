FROM satijalab/seurat:latest

RUN R -e "install.packages(c('optparse', 'devtools'), dependencies=TRUE)" && \
    R -e "devtools::install_github('haotian-zhuang/findPC')"

COPY /seurat_fine_mapping/calculate_phi_k.R /