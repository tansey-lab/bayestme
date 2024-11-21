import logging
import re
from enum import Enum

import anndata
import numpy as np
import pandas

from . import data, utils

logger = logging.getLogger(__name__)


class FilterType(Enum):
    SPOTS = 1
    RIBOSOME = 2


def select_top_genes_by_standard_deviation(
    dataset: data.SpatialExpressionDataset, n_gene: int
) -> data.SpatialExpressionDataset:
    # order genes by the standard deviation across spots
    ordering = utils.get_stddev_ordering(dataset.counts)

    n_top_genes = min(n_gene, dataset.n_gene)

    logger.info(
        "filtering top {} genes from original {} genes...".format(
            n_top_genes, dataset.n_gene
        )
    )
    n_gene_filter = ordering[:n_top_genes]

    input_adata = dataset.adata

    return data.SpatialExpressionDataset(input_adata[:, n_gene_filter])


def filter_genes_by_spot_threshold(
    dataset: data.SpatialExpressionDataset, spot_threshold: float
):
    n_spots = dataset.counts.shape[0]

    keep = (dataset.counts > 0).sum(axis=0) <= int(n_spots * spot_threshold)

    input_adata = dataset.adata
    return data.SpatialExpressionDataset(input_adata[:, keep])


RIBOSOME_GENE_NAME_PATTERN = "[Rr][Pp][SsLl]"


def filter_ribosome_genes(dataset: data.SpatialExpressionDataset):
    keep = ~np.array(
        [bool(re.match(RIBOSOME_GENE_NAME_PATTERN, g)) for g in dataset.gene_names],
        dtype=bool,
    )

    input_adata = dataset.adata
    return data.SpatialExpressionDataset(input_adata[:, keep])


def filter_list_of_genes(dataset: data.SpatialExpressionDataset, genes_to_remove):
    keep = ~np.array([g in genes_to_remove for g in dataset.gene_names], dtype=bool)

    input_adata = dataset.adata
    return data.SpatialExpressionDataset(input_adata[:, keep])


def filter_stdata_to_match_expression_truth(
    stdata: data.SpatialExpressionDataset, adata_fn: str
):
    """
    Filter the stdata down to the intersection of genes between it and the expression truth file.

    :param stdata: SpatialExpressionDataset object
    :param adata_fn: h5ad file with matched scRNA
    :return: Filtered stdata object
    """
    ad = anndata.read_h5ad(adata_fn)

    genes_in_st_but_not_in_scrna = set(stdata.gene_names.tolist()).difference(
        ad.var_names
    )

    return filter_list_of_genes(stdata, genes_in_st_but_not_in_scrna)
