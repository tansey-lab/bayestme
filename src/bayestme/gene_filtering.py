import numpy as np
import re
from enum import Enum
import logging

from . import data, utils

logger = logging.getLogger(__name__)


class FilterType(Enum):
    SPOTS = 1
    RIBOSOME = 2


def select_top_genes_by_standard_deviation(
        dataset: data.SpatialExpressionDataset,
        n_gene: int) -> data.SpatialExpressionDataset:
    # order genes by the standard deviation across spots
    ordering = utils.get_stddev_ordering(dataset.reads)

    n_top_genes = min(n_gene, dataset.n_gene)

    logger.info('filtering top {} genes from original {} genes...'.format(n_top_genes, dataset.n_gene))
    n_gene_filter = ordering[:n_top_genes]
    filtered_raw_counts = dataset.raw_counts[:, n_gene_filter]
    filtered_gene_names = dataset.gene_names[n_gene_filter]

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=filtered_raw_counts,
        positions=dataset.positions,
        tissue_mask=dataset.tissue_mask,
        gene_names=filtered_gene_names,
        layout=dataset.layout
    )


def filter_genes_by_spot_threshold(dataset: data.SpatialExpressionDataset, spot_threshold: float):
    n_spots = dataset.reads.shape[0]

    keep = (dataset.reads > 0).sum(axis=0) < int(n_spots * spot_threshold)

    filtered_raw_counts = dataset.raw_counts[:, keep]
    filtered_gene_names = dataset.gene_names[keep]

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=filtered_raw_counts,
        positions=dataset.positions,
        tissue_mask=dataset.tissue_mask,
        gene_names=filtered_gene_names,
        layout=dataset.layout
    )


RIBOSOME_GENE_NAME_PATTERN = '[Rr][Pp][SsLl]'


def filter_ribosome_genes(dataset: data.SpatialExpressionDataset):
    keep = ~np.array([bool(re.match(RIBOSOME_GENE_NAME_PATTERN, g)) for g in dataset.gene_names], dtype=np.bool)

    filtered_raw_counts = dataset.raw_counts[:, keep]
    filtered_gene_names = dataset.gene_names[keep]

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=filtered_raw_counts,
        positions=dataset.positions,
        tissue_mask=dataset.tissue_mask,
        gene_names=filtered_gene_names,
        layout=dataset.layout
    )
