import numpy as np
import pandas

from bayestme import data


def rescale_coordinate_to_integers(coordinate):
    coordinate_sorted = np.sort(coordinate.unique())
    return coordinate.apply(lambda x: np.where(coordinate_sorted == x)[0][0])


def rescale_and_round_data(arr):
    signal_maxes = arr.max(axis=0)

    rescaled_counts = (arr * 100) / signal_maxes

    return rescaled_counts.round().astype(int)


def maldi_dataframe_to_stdata(df: pandas.DataFrame) -> data.SpatialExpressionDataset:
    barcodes = df['Spot'].to_numpy()
    features = df.columns[3:].to_numpy()

    n_spot = len(barcodes)

    tissue_mask = np.array([True]*n_spot, dtype=bool)

    raw_counts = df[df.columns[3:]].to_numpy()
    signal_maxes = raw_counts.max(axis=0)
    nonzero_signals = signal_maxes != 0
    raw_counts_minus_bad_signals = raw_counts[:, nonzero_signals]

    integer_rescaled_counts = rescale_and_round_data(raw_counts_minus_bad_signals)

    positions = np.vstack(
        [rescale_coordinate_to_integers(df.x).to_numpy(),
         rescale_coordinate_to_integers(df.y).to_numpy()]
    ).T

    return data.SpatialExpressionDataset.from_arrays(
        raw_counts=integer_rescaled_counts,
        positions=positions,
        tissue_mask=tissue_mask,
        gene_names=features[nonzero_signals],
        layout=data.Layout.SQUARE,
        barcodes=barcodes)
