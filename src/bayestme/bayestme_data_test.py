import tempfile
import os.path
import pandas
import pytest
import glob
import numpy as np

from bayestme.bayestme import BayesTME
from bayestme import bayestme_data, bleeding_correction

TEST_DATA = [
    {'gene': 'PSME2 ENSG00000100911', '7x15': 2, '7x16': 0, '7x17': 0, '7x18': 0, '8x13': 0, '8x14': 1, '8x15': 3,
     '8x16': 1, '8x17': 4},
    {'gene': 'CUEDC1 ENSG00000180891', '7x15': 0, '7x16': 0, '7x17': 0, '7x18': 1, '8x13': 0, '8x14': 0, '8x15': 0,
     '8x16': 0, '8x17': 1},
    {'gene': 'RPLP1 ENSG00000137818', '7x15': 21, '7x16': 13, '7x17': 13, '7x18': 18, '8x13': 10, '8x14': 29,
     '8x15': 45, '8x16': 8, '8x17': 52},
    {'gene': 'TM9SF3 ENSG00000077147', '7x15': 0, '7x16': 0, '7x17': 0, '7x18': 0, '8x13': 0, '8x14': 1, '8x15': 1,
     '8x16': 0, '8x17': 1},
    {'gene': 'DEF8 ENSG00000140995', '7x15': 0, '7x16': 0, '7x17': 0, '7x18': 1, '8x13': 0, '8x14': 2, '8x15': 1,
     '8x16': 0, '8x17': 0}]


def test_clean_bleed_doesnt_run_without_non_tissue_spots():
    tmp_dir = tempfile.mkdtemp()

    pandas.DataFrame(TEST_DATA).to_csv(os.path.join(tmp_dir, 'data.tsv'), sep="\t", index=False, header=True)

    reader = BayesTME(storage_path=tmp_dir)
    stdata = reader.load_data_from_count_mat(os.path.join(tmp_dir, 'data.tsv'))

    cleaned_stdata = bayestme_data.CleanedSTData(
        stdata=stdata,
        n_top=3,
        max_steps=5)
    cleaned_stdata.load_data('test')

    assert not cleaned_stdata.has_non_tissue_spots()

    with pytest.raises(RuntimeError) as excinfo:
        cleaned_stdata.clean_bleed()


def test_clean_bleed():
    tmp_dir = tempfile.mkdtemp()

    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=12,
        n_cols=12,
        n_genes=5)

    stdata = bayestme_data.RawSTData(
        data_name='test', load=None, raw_count=bleed_counts, positions=locations.T, tissue_mask=tissue_mask, gene_names=np.array(['1', '2', '3']),
        layout=2, storage_path=tmp_dir,
    )

    cleaned_stdata = bayestme_data.CleanedSTData(
        stdata=stdata,
        n_top=3,
        max_steps=5)

    cleaned_stdata.load_data('fsdf')

    cleaned_stdata.clean_bleed(n_top=3)


def test_cv_prepare_jobs():
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=12,
        n_cols=12,
        n_genes=3)
    tmp_dir = tempfile.mkdtemp()

    stdata = bayestme_data.RawSTData(
        'test', load=False, raw_count=bleed_counts, positions=locations.T, tissue_mask=tissue_mask,
        gene_names=[str(x) for x in range(3)],
        layout=2,
        storage_path=tmp_dir,
        x_y_swap=False,
        invert=[0, 0])

    cv_stdata = bayestme_data.CrossValidationSTData(stdata=stdata,
                                                    n_fold=1,
                                                    n_splits=2,
                                                    n_samples=1,
                                                    n_burn=3,
                                                    n_thin=3,
                                                    lda=0,
                                                    n_comp_min=2,
                                                    n_comp_max=3,
                                                    lambda_values=(1, 1e1, 1e2),
                                                    max_ncell=120)

    cv_stdata.prepare_jobs()

    assert len(list(glob.glob(os.path.join(tmp_dir, 'k_fold/setup/config/test/*.cfg')))) == 6


def test_create_folds():
    locations, tissue_mask, true_rates, true_counts, bleed_counts = bleeding_correction.generate_data(
        n_rows=15,
        n_cols=15,
        n_genes=3)

    folds = [x for x in bayestme_data.CrossValidationSTData.create_folds(
        n_spot_in=tissue_mask.sum(),
        positions_tissue=locations.T[:, tissue_mask],
        layout=2,
        reads=bleed_counts[tissue_mask],
        n_fold=5,
        n_splits=5
    )]
