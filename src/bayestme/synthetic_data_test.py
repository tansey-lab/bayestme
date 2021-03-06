from bayestme import synthetic_data, data


def test_generate_fake_stdataset():
    n_rows = 20
    n_cols = 20
    sq = synthetic_data.generate_fake_stdataset(n_genes=1, n_cols=n_cols, n_rows=n_rows, layout=data.Layout.SQUARE)
    hex = synthetic_data.generate_fake_stdataset(n_genes=1, n_cols=n_cols, n_rows=n_rows, layout=data.Layout.HEX)

    assert sq.positions.shape == hex.positions.shape
