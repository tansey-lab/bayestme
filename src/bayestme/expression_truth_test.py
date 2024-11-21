import os
import tempfile
import anndata
import shutil
import numpy as np
import pandas

import bayestme.common
import bayestme.data
import bayestme.expression_truth
import bayestme.synthetic_data
import bayestme.utils
from bayestme import expression_truth, data


def test_combine_multiple_expression_truth():
    data = [
        np.array([[1, 2, 3, 0], [2, 3, 4, 1], [2, 2, 2, 0]]),
        np.array([[1, 2, 4, 0], [2, 3, 6, 1], [2, 1, 2, 0]]),
    ]

    result = expression_truth.combine_multiple_expression_truth(
        expression_truth_arrays=data, num_warmup=10, num_samples=10
    )

    assert result.shape == (3, 4)
    assert np.all(result > 0)


def test_load_phi_truth():
    example_data = {
        "0": {
            "LINC01409": 6.20854181530519e-06,
            "LINC01128": 7.94814125442517e-06,
            "LINC00115": 4.54911539806641e-06,
            "FAM41C": 0.0,
            "SAMD11": 0.0,
        },
        "1": {
            "LINC01409": 1.49995044600798e-05,
            "LINC01128": 2.35076069675239e-05,
            "LINC00115": 2.27916942034948e-06,
            "FAM41C": 0.0,
            "SAMD11": 1.40427550443168e-06,
        },
        "2": {
            "LINC01409": 2.97631602698784e-06,
            "LINC01128": 1.45160772961017e-05,
            "LINC00115": 1.46704940943111e-06,
            "FAM41C": 7.18370422770737e-07,
            "SAMD11": 0.0,
        },
        "3": {
            "LINC01409": 7.58645413781156e-06,
            "LINC01128": 8.1726936453575e-06,
            "LINC00115": 6.12543772975621e-07,
            "FAM41C": 0.0,
            "SAMD11": 0.0,
        },
        "4": {
            "LINC01409": 1.87487192566961e-06,
            "LINC01128": 1.70604741811557e-05,
            "LINC00115": 3.46678185392759e-06,
            "FAM41C": 0.0,
            "SAMD11": 7.77426624554906e-07,
        },
        "5": {
            "LINC01409": 7.83310236784797e-06,
            "LINC01128": 9.96282875190567e-06,
            "LINC00115": 4.87937499885119e-06,
            "FAM41C": 9.4615520015824e-07,
            "SAMD11": 0.0,
        },
        "6": {
            "LINC01409": 6.68166666509352e-06,
            "LINC01128": 2.59127189953248e-05,
            "LINC00115": 0.0,
            "FAM41C": 0.0,
            "SAMD11": 0.0,
        },
        "7": {
            "LINC01409": 3.87608783680169e-06,
            "LINC01128": 1.35800435754236e-05,
            "LINC00115": 9.97461460582817e-06,
            "FAM41C": 0.0,
            "SAMD11": 0.0,
        },
        "8": {
            "LINC01409": 1.05828228135006e-05,
            "LINC01128": 4.36250687094832e-06,
            "LINC00115": 4.72581079234104e-06,
            "FAM41C": 0.0,
            "SAMD11": 0.0,
        },
    }

    tmpdir = tempfile.mkdtemp()

    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=12, n_cols=12, n_genes=3
    )

    dataset = data.SpatialExpressionDataset.from_arrays(
        raw_counts=bleed_counts,
        tissue_mask=tissue_mask,
        positions=locations,
        gene_names=np.array(["LINC01409", "LINC01128", "LINC00115"]),
        layout=bayestme.common.Layout.SQUARE,
        edges=bayestme.utils.get_edges(
            locations[tissue_mask], bayestme.common.Layout.SQUARE
        ),
    )

    pandas.DataFrame(example_data).to_csv(os.path.join(tmpdir, "test.csv"), index=True)

    result = bayestme.expression_truth.load_expression_truth(
        dataset, os.path.join(tmpdir, "test.csv")
    )

    assert result.shape == (9, 3)

    np.testing.assert_almost_equal(result.sum(axis=1), np.ones(9))


def test_calculate_celltype_profile_prior_from_adata():
    tmpdir = tempfile.mkdtemp()
    ad_path = os.path.join(tmpdir, "data.h5")

    ad = anndata.AnnData(
        X=np.array([[1, 2, 3], [2, 3, 4], [2, 2, 2], [1, 2, 3]]),
        obs=pandas.DataFrame(
            {"celltype": ["a", "b", "a", "b"], "sample": ["1", "1", "2", "2"]}
        ),
    )

    ad.var_names = ["a", "b", "c"]
    ad.obs_names = ["1", "2", "3", "4"]
    ad.write_h5ad(ad_path)
    result = bayestme.expression_truth.calculate_celltype_profile_prior_from_adata(
        ad_path,
        gene_names=["b", "c", "a"],
        celltype_column="celltype",
        sample_column="sample",
    )

    assert result.shape == (2, 3)

    ad = anndata.AnnData(
        X=np.array([[1, 2, 3], [2, 3, 4], [2, 2, 2], [1, 2, 3]]),
        obs=pandas.DataFrame(
            {"celltype": ["a", "b", "a", "b"], "sample": ["1", "1", "1", "1"]}
        ),
    )

    ad.var_names = ["a", "b", "c"]
    ad.obs_names = ["1", "2", "3", "4"]
    ad.write_h5ad(ad_path)
    result = bayestme.expression_truth.calculate_celltype_profile_prior_from_adata(
        ad_path,
        gene_names=["b", "c", "a"],
        celltype_column="celltype",
        sample_column="sample",
    )

    assert result.shape == (2, 3)
    np.testing.assert_almost_equal(
        result,
        np.array([[0.33333333, 0.4, 0.26666667], [0.33333333, 0.44444444, 0.22222222]]),
    )
    ad.write_h5ad(ad_path)
    result = bayestme.expression_truth.calculate_celltype_profile_prior_from_adata(
        ad_path,
        gene_names=["c", "b", "a"],
        celltype_column="celltype",
        sample_column="sample",
    )

    assert result.shape == (2, 3)
    np.testing.assert_almost_equal(
        result,
        np.array([[0.4, 0.33333333, 0.26666667], [0.44444444, 0.33333333, 0.22222222]]),
    )
    shutil.rmtree(tmpdir)
