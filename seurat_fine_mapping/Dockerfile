FROM satijalab/seurat:latest

RUN R -e "install.packages(c('optparse', 'devtools'), dependencies=TRUE)" && \
    R -e "devtools::install_github('haotian-zhuang/findPC')"

COPY calculate_phi_k.R /