import numpy as np

from bayestme import gene_filtering, data, bleeding_correction


def test_select_top_genes_by_standard_deviation():
    # Given: Three spots and three genes, the first two of which have large variation in counts
    raw_counts = np.array(
        [[199, 200, 1],
         [10000, 10001, 2],
         [1, 3, 3]], dtype=np.int64
    )

    locations = np.array([
        (x, 0) for x in range(3)
    ])

    tissue_mask = np.array([True for _ in range(3)])

    gene_names = np.array(['keep_me1',
                           'keep_me2',
                           'filter'])

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=gene_names,
        layout=data.Layout.SQUARE
    )

    # When: select_top_genes_by_standard_deviation is called to select the top 2 genes
    n_genes_filter = 2
    result = gene_filtering.select_top_genes_by_standard_deviation(dataset, n_genes_filter)

    (n_spots_in, n_genes) = result.reads.shape

    # Then: the 2 high variation genes are selected
    assert n_genes == n_genes_filter
    assert n_spots_in == 3
    np.testing.assert_equal(
        result.reads,
        np.array(
            [[199, 200],
             [10000, 10001],
             [1, 3]], dtype=np.int64
        )
    )
    np.testing.assert_equal(result.gene_names, np.array(['keep_me1', 'keep_me2']))


def test_filter_genes_by_spot_threshold():
    # Given: Three spots and three genes, the first two of which appear in 66% of spots, the third gene never appears
    raw_counts = np.array(
        [[1, 1, 0],
         [1, 1, 0],
         [0, 0, 0]], dtype=np.int64
    )

    locations = np.array([
        (x, 0) for x in range(3)
    ])

    tissue_mask = np.array([True for _ in range(3)])

    gene_names = np.array(['filter1',
                           'filter2',
                           'keep_me'])

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=gene_names,
        layout=data.Layout.SQUARE
    )

    # When: filter_genes_by_spot_threshold is called with threshold 0.5

    result = gene_filtering.filter_genes_by_spot_threshold(dataset, spot_threshold=0.5)

    (n_spots_in, n_genes) = result.reads.shape

    # Then: only the gene which appears in 0% of spots is kept
    assert n_genes == 1
    np.testing.assert_equal(
        result.reads,
        np.array(
            [[0],
             [0],
             [0]], dtype=np.int64
        )
    )
    np.testing.assert_equal(result.gene_names, np.array(['keep_me']))


def test_filter_ribosome_genes():
    # Given: Three genes, one of which has a name that matches the ribosome gene name pattern
    raw_counts = np.array(
        [[7, 1, 2],
         [8, 2, 3],
         [9, 3, 4]], dtype=np.int64
    )

    locations = np.array([
        (x, 0) for x in range(3)
    ])

    tissue_mask = np.array([True for _ in range(3)])

    gene_names = np.array(['RPL333',
                           'other',
                           'other2'])
    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=gene_names,
        layout=data.Layout.SQUARE
    )

    # When: filter_ribosome_genes is called
    result = gene_filtering.filter_ribosome_genes(dataset)

    (n_spots_in, n_genes) = result.reads.shape

    # Then: only the two genes with non matching names are kept
    assert n_genes == 2
    np.testing.assert_equal(
        result.reads,
        np.array(
            [[1, 2],
             [2, 3],
             [3, 4]], dtype=np.int64
        )
    )
    np.testing.assert_equal(result.gene_names, np.array(['other', 'other2']))