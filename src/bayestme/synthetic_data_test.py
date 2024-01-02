import bayestme.common
from bayestme import synthetic_data, data


def test_generate_demo_dataset():
    synthetic_data.generate_demo_dataset()
    synthetic_data.generate_demo_dataset_with_bleeding()


def test_generate_fake_stdataset():
    n_rows = 20
    n_cols = 20
    sq = synthetic_data.generate_fake_stdataset(
        n_genes=1, n_cols=n_cols, n_rows=n_rows, layout=bayestme.common.Layout.SQUARE
    )
    hex = synthetic_data.generate_fake_stdataset(
        n_genes=1, n_cols=n_cols, n_rows=n_rows, layout=bayestme.common.Layout.HEX
    )

    assert sq.positions.shape == hex.positions.shape
