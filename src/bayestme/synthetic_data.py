import numpy as np
from scipy.stats import multivariate_normal, multivariate_t

import bayestme.data
import bayestme.marker_genes

from bayestme import data


def generate_simulated_bleeding_reads_data(
    n_rows=30,
    n_cols=30,
    n_genes=20,
    spot_bleed_prob=0.5,
    length_scale=0.2,
    gene_bandwidth=1,
    bleeding="anisotropic",
):
    """
    Generate simulated read data with modeled bleeding.

    :param n_rows: Number of spot rows
    :param n_cols: Number of spot columns
    :param n_genes: Number of genes in output reads dataset
    :param spot_bleed_prob: Tuning param
    :param length_scale: Tuning param
    :param gene_bandwidth: Tuning param
    :param bleeding: Type of bleeding
    :return: (locations, tissue_mask, true_rates, true_counts, bleed_counts)
    """
    xygrid = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
    locations = np.array([xygrid[0].reshape(-1), xygrid[1].reshape(-1)]).T

    # In-tissue region is the central half
    tissue_mask = (
        (locations[:, 0] > n_rows / 4)
        & (locations[:, 0] < n_rows / 4 * 3)
        & (locations[:, 1] > n_cols / 4)
        & (locations[:, 1] < n_cols / 4 * 3)
    )

    # Sample the true gene reads
    true_rates = np.zeros((n_rows * n_cols, n_genes))
    true_rates[tissue_mask] = np.random.gamma(20, 10, size=(1, n_genes))

    # Make the genes vary in space, except gene 1 which is a control example
    Cov = length_scale * np.exp(
        -np.array(
            [
                ((l[None] - locations[tissue_mask]) ** 2).sum(axis=-1)
                for l in locations[tissue_mask]
            ]
        )
        / (2 * gene_bandwidth**2)
    ) + np.diag(np.ones(tissue_mask.sum()) * 1e-4)

    for g in range(1, n_genes):
        true_rates[tissue_mask, g] *= np.exp(
            np.random.multivariate_normal(np.zeros(tissue_mask.sum()), Cov)
        )

        # Insert some regions of sparsity
        start = np.array([n_rows / 2, n_cols / 2])

        # Add a random offset
        start = np.round(
            start
            + (np.random.random(size=2) * 2 - 1) * np.array([n_rows / 4, n_cols / 4])
        ).astype(int)

        # Draw a box of sparsity
        width = n_rows // 6
        height = n_cols // 6
        sparsity_mask = (
            (locations[:, 0] >= start[0])
            & (locations[:, 0] < start[0] + width)
            & (locations[:, 1] >= start[1])
            & (locations[:, 1] < start[1] + height)
        )

        true_rates[sparsity_mask, g] = 0

    true_counts = np.random.poisson(true_rates * spot_bleed_prob)

    # Add some anisotropic bleeding
    bleed_counts = np.zeros_like(true_counts)
    if bleeding == "gaussian":
        x, y = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
        pos = np.dstack((x, y))
        for i in range(tissue_mask.sum()):
            x_cor, y_cor = locations[tissue_mask][i]
            rv_gaus = multivariate_normal([x_cor, y_cor], [[5, 1], [1, 5]])
            for g in range(n_genes):
                bleed_counts[:, g] += np.random.multinomial(
                    true_counts[tissue_mask][i, g], rv_gaus.pdf(pos).flatten()
                )
    elif bleeding == "t":
        x, y = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
        pos = np.dstack((x, y))
        for i in range(tissue_mask.sum()):
            x_cor, y_cor = locations[tissue_mask][i]
            rv_t = multivariate_t([x_cor, y_cor], [[20, 3], [3, 30]], df=10)
            for g in range(n_genes):
                bleed_counts[:, g] += np.random.multinomial(
                    true_counts[tissue_mask][i, g], rv_t.pdf(pos).flatten()
                )
    elif bleeding == "anisotropic":
        Distances = np.zeros((n_rows * n_cols, n_rows * n_cols, 4))
        true_w = np.array([0.2, 0.03, 1.5, 0.05])
        true_BleedProbs = np.zeros((n_rows * n_cols, n_rows * n_cols))
        for i in range(n_rows * n_cols):
            if i % 100 == 0:
                print(i)
            Distances[:, i, 0] = (locations[i, 0] - locations[:, 0]).clip(0, None) ** 2
            Distances[:, i, 1] = (locations[:, 0] - locations[i, 0]).clip(0, None) ** 2
            Distances[:, i, 2] = (locations[i, 1] - locations[:, 1]).clip(0, None) ** 2
            Distances[:, i, 3] = (locations[:, 1] - locations[i, 1]).clip(0, None) ** 2
            h = np.exp(-Distances[:, i].dot(true_w))
            true_BleedProbs[:, i] = h / h.sum()
            for g in range(n_genes):
                bleed_counts[:, g] += np.random.multinomial(
                    true_counts[i, g], true_BleedProbs[:, i]
                )

    # Add the counts due to non-bleeding
    local_counts = np.random.poisson(true_rates * (1 - spot_bleed_prob))
    true_counts += local_counts
    bleed_counts += local_counts

    return locations, tissue_mask, true_rates, true_counts, bleed_counts


def generate_fake_stdataset(
    n_rows: int = 30,
    n_cols: int = 30,
    n_genes: int = 20,
    layout: data.Layout = data.Layout.SQUARE,
) -> data.SpatialExpressionDataset:
    """
    Create a fake dataset for use in testing or demonstration.

    :param n_rows: width of the fake slide
    :param n_cols: height of the fake slide
    :param n_genes: number of marker genes
    :param layout: layout of spots on the fake slide
    :return: SpatialExpressionDataset object containing simulated data
    """
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = generate_simulated_bleeding_reads_data(
        n_rows=n_rows, n_cols=n_cols, n_genes=n_genes
    )

    if layout is data.Layout.HEX:
        locations[:, 1] = locations[:, 1] * 2
        locations[locations[:, 0] % 2 == 1, 1] += 1
    elif layout is data.Layout.SQUARE:
        locations = locations
    else:
        raise NotImplementedError(layout)

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["{}".format(x) for x in range(n_genes)]),
        layout=layout,
        barcodes=np.array(["barcode" + str(i) for i in range(len(locations))]),
    )


def generate_demo_dataset():
    """
    Generate a fake dataset for use in testing or demonstration.
    This dataset should produce two cell types, with 2 markers genes for each cell type.
    """
    slide_tissue = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    gene_north_exp = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 9, 0, 0, 0, 0],
            [0, 0, 0, 7, 7, 7, 0, 0, 0],
            [0, 0, 3, 4, 5, 4, 3, 0, 0],
            [0, 1, 2, 3, 4, 3, 2, 1, 0],
            [0, 0, 1, 2, 3, 2, 1, 0, 0],
            [0, 0, 0, 1, 2, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    gene_south_exp = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 1, 0, 0, 0],
            [0, 0, 1, 2, 3, 2, 1, 0, 0],
            [0, 1, 2, 3, 4, 3, 2, 1, 0],
            [0, 0, 3, 4, 5, 4, 3, 0, 0],
            [0, 0, 0, 7, 7, 7, 0, 0, 0],
            [0, 0, 0, 0, 9, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    noise_exp = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 1, 0, 0, 0],
            [0, 0, 1, 2, 1, 2, 1, 0, 0],
            [0, 1, 2, 1, 2, 1, 2, 1, 0],
            [0, 0, 3, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    tissue_mask = slide_tissue.flatten() == 1

    raw_counts = np.vstack(
        [
            gene_north_exp.flatten(),
            gene_north_exp.flatten() * 10,
            gene_south_exp.flatten(),
            gene_south_exp.flatten() * 10,
            noise_exp.flatten(),
            noise_exp.flatten() * 2,
        ]
    ).T

    locations = np.vstack([np.repeat(np.arange(9), 9), np.tile(np.arange(9), 9)]).T

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=raw_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(
            [
                "north_weak",
                "north_strong",
                "south_weak",
                "south_strong",
                "noise_weak",
                "noise_strong",
            ]
        ),
        layout=data.Layout.SQUARE,
        barcodes=np.array(["barcode" + str(i) for i in range(len(locations))]),
    )


def create_deconvolve_dataset(
    n_nodes: int = 12,
    n_components: int = 5,
    n_samples: int = 100,
    n_genes: int = 100,
    n_marker_gene: int = 5,
):
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=n_nodes, n_cols=n_nodes, n_genes=n_genes
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["gene{}".format(x) for x in range(n_genes)]),
        layout=data.Layout.SQUARE,
    )

    deconvolve_results = create_toy_deconvolve_result(
        n_nodes=dataset.n_spot_in,
        n_components=n_components,
        n_samples=n_samples,
        n_gene=dataset.n_gene,
    )

    bayestme.data.add_deconvolution_results_to_dataset(
        stdata=dataset, result=deconvolve_results
    )

    marker_genes = bayestme.marker_genes.select_marker_genes(
        deconvolution_result=deconvolve_results, n_marker=n_marker_gene, alpha=0.99
    )

    bayestme.marker_genes.add_marker_gene_results_to_dataset(
        stdata=dataset, marker_genes=marker_genes
    )

    return dataset


def create_toy_deconvolve_result(
    n_nodes: int, n_components: int, n_samples: int, n_gene: int
) -> data.DeconvolutionResult:
    return data.DeconvolutionResult(
        lam2=1000,
        n_components=n_components,
        cell_num_trace=np.random.random((n_samples, n_nodes, n_components + 1)),
        cell_prob_trace=np.random.random((n_samples, n_nodes, n_components + 1)),
        expression_trace=np.random.random((n_samples, n_components, n_gene)),
        beta_trace=np.random.random((n_samples, n_components)),
        reads_trace=np.random.random((n_samples, n_nodes, n_gene, n_components)),
    )
