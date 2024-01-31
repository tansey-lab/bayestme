import anndata
import numpy as np

from bayestme import semi_synthetic_spatial


def test_semi_synthetic_spatial():
    counts = np.random.poisson(30, size=100 * 100).reshape((100, 100))

    ad = anndata.AnnData(X=counts)

    ad.obs["cluster"] = np.random.choice(np.array(["1", "2", "3"]), size=100)
    a, b = np.meshgrid(np.arange(6, 30), np.arange(6, 30))
    pos_ss = np.column_stack((a.ravel(), b.ravel()))
    n_genes = 20
    (
        stdata,
        Truth_prior,
        n_cells,
        spatial,
        sampled_cell_reads,
    ) = semi_synthetic_spatial.generate_semi_synthetic(
        ad, "cluster", pos_ss, n_genes=n_genes, canvas_size=(36, 36), n_spatial_gene=5
    )

    assert stdata.counts.shape == (pos_ss.shape[0], n_genes)
