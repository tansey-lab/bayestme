import numpy as np
import pandas as pd
import scipy.io as io
import os
import glob
import logging
import h5py
import anndata
from scipy.sparse import csr_matrix

from enum import Enum
from typing import Optional

from . import utils

logger = logging.getLogger(__name__)


IN_TISSUE_ATTR = 'in_tissue'
SPATIAL_ATTR = 'spatial'
LAYOUT_ATTR = 'layout'


class Layout(Enum):
    HEX = 1
    SQUARE = 2


def create_anndata_object(
        counts: np.ndarray,
        coordinates: Optional[np.ndarray],
        tissue_mask: Optional[np.ndarray],
        gene_names: np.ndarray,
        layout: Layout):

    coordinates = coordinates.astype(int)
    adata = anndata.AnnData(counts, obsm={SPATIAL_ATTR: coordinates})
    adata.obs[IN_TISSUE_ATTR] = tissue_mask
    adata.uns[LAYOUT_ATTR] = layout.value
    adata.var_names = gene_names

    return adata


class SpatialExpressionDataset:
    """
    Data model for holding read counts, their associated position information,
    and whether they come from tissue or non tissue spots.
    Also holds the names of the gene markers in the dataset.
    """
    def __init__(self,
                 raw_counts: np.ndarray,
                 positions: Optional[np.ndarray],
                 tissue_mask: Optional[np.ndarray],
                 gene_names: np.ndarray,
                 layout: Layout):
        """
        :param raw_counts: An <N spots> x <N markers> matrix.
        :param positions: An <N spots> x 2 matrix of spot coordinates.
        :param tissue_mask: An <N spot> length array of booleans. True if spot is in tissue, False if not.
        :param gene_names: An <M markers> length array of gene names.
        :param layout: Layout.SQUARE of the spots are in a square grid layout, Layout.HEX if the spots are
        in a hex grid layout.
        """
        self.layout = layout
        self.gene_names = gene_names
        self.tissue_mask = tissue_mask
        self.positions = positions.astype(int)
        self.raw_counts = raw_counts
        self.edges = utils.get_edges(self.positions_tissue, layout=self.layout.value)

    @property
    def reads(self) -> np.ndarray:
        return self.raw_counts[self.tissue_mask]

    @property
    def positions_tissue(self) -> np.ndarray:
        return self.positions[self.tissue_mask, :]

    @property
    def n_spot_in(self) -> int:
        return self.tissue_mask.sum()

    @property
    def n_gene(self) -> int:
        return self.raw_counts.shape[1]

    def save(self, path):
        with h5py.File(path, 'w') as f:
            f['raw_counts'] = self.raw_counts
            f['positions'] = self.positions
            f['tissue_mask'] = self.tissue_mask
            f['gene_names'] = self.gene_names.astype('S')
            f.attrs['layout'] = self.layout.name

    @classmethod
    def read_spaceranger(cls, data_path, layout=Layout.HEX):
        """
        Load data from spaceranger /outputs folder

        :param data_path: Directory containing at least
            1) /raw_feature_bc_matrix for raw count matrix
            2) /filtered_feature_bc_matrix for filtered count matrix
            3) /spatial for position list
        :param layout: Layout.SQUARE of the spots are in a square grid layout, Layout.HEX if the spots are
        in a hex grid layout.
        :return: SpatialExpressionDataset
        """
        raw_count_path = os.path.join(data_path, 'raw_feature_bc_matrix/matrix.mtx.gz')
        filtered_count_path = os.path.join(data_path, 'filtered_feature_bc_matrix/matrix.mtx.gz')
        features_path = os.path.join(data_path, 'raw_feature_bc_matrix/features.tsv.gz')
        barcodes_path = os.path.join(data_path, 'raw_feature_bc_matrix/barcodes.tsv.gz')
        positions_path = glob.glob(os.path.join(data_path, 'spatial/tissue_positions_list.*')).pop()

        positions_list = pd.read_csv(positions_path, header=None, index_col=0, names=None)

        raw_count = np.array(io.mmread(raw_count_path).todense())
        filtered_count = np.array(io.mmread(filtered_count_path).todense())
        features = np.array(pd.read_csv(features_path, header=None, sep='\t'))[:, 1].astype(str)
        barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')
        n_spots = raw_count.shape[1]
        n_genes = raw_count.shape[0]
        logger.info('detected {} spots, {} genes'.format(n_spots, n_genes))
        pos = np.zeros((n_spots, 3))
        for i in range(n_spots):
            pos[i] = np.array(positions_list.loc[barcodes[0][i]][:3])
        tissue_mask = pos[:, 0] == 1
        positions = pos[:, 1:]
        n_spot_in = tissue_mask.sum()
        logger.info('\t {} spots in tissue sample'.format(n_spot_in))
        all_counts = raw_count.sum()
        tissue_counts = filtered_count.sum()
        logger.info('\t {:.3f}% UMI counts bleeds out'.format((1 - tissue_counts / all_counts) * 100))

        return cls(
            raw_counts=raw_count.T,
            positions=positions.T,
            tissue_mask=tissue_mask,
            gene_names=features,
            layout=layout
        )

    @classmethod
    def read_count_mat(cls, data_path, layout=Layout.SQUARE):
        """
        Load data from tsv count matrix containing only in-tissue spots where the count matrix
        is a tsv file of shape G by N
        The column names and row names are position and gene names respectively

        :param data_path: /path/to/count_matrix
        :param layout: Layout.SQUARE of the spots are in a square grid layout, Layout.HEX if the spots are
        in a hex grid layout.
        :return: SpatialExpressionDataset
        """
        raw_data = pd.read_csv(data_path, sep='\t')
        count_mat = raw_data.values[:, 1:].T.astype(int)
        features = np.array([x.split(' ')[0] for x in raw_data.values[:, 0]])
        n_spots = count_mat.shape[0]
        n_genes = count_mat.shape[1]
        logger.info('detected {} spots, {} genes'.format(n_spots, n_genes))
        positions = np.zeros((2, n_spots))
        for i in range(n_spots):
            spot_pos = raw_data.columns[1:][i].split('x')
            positions[0, i] = int(spot_pos[0])
            positions[1, i] = int(spot_pos[1])
        positions = positions.astype(int)
        tissue_mask = np.ones(n_spots).astype(bool)

        return cls(
            raw_counts=count_mat,
            positions=positions,
            tissue_mask=tissue_mask,
            gene_names=features,
            layout=layout)

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, 'r') as f:
            raw_counts = f['raw_counts'][:]
            positions = f['positions'][:]
            tissue_mask = f['tissue_mask'][:]
            gene_names = np.array([x.decode('utf-8') for x in f['gene_names'][:]])
            layout_name = f.attrs['layout']
            layout = Layout[layout_name]

            return cls(
                raw_counts=raw_counts,
                positions=positions,
                tissue_mask=tissue_mask,
                gene_names=gene_names,
                layout=layout)


class BleedCorrectionResult:
    """
    Data model for the results of bleeding correction.
    """
    def __init__(self,
                 corrected_reads: np.ndarray,
                 global_rates: np.ndarray,
                 basis_functions: np.ndarray,
                 weights: np.ndarray):
        """
        :param corrected_reads: <N in-tissue spot> x <N genes> matrix of corrected read counts.
        :param global_rates:
        :param basis_functions:
        :param weights:
        """
        self.weights = weights
        self.basis_functions = basis_functions
        self.global_rates = global_rates
        self.corrected_reads = corrected_reads

    def save(self, path):
        with h5py.File(path, 'w') as f:
            f['corrected_reads'] = self.corrected_reads
            f['global_rates'] = self.global_rates
            f['basis_functions'] = self.basis_functions
            f['weights'] = self.weights

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, 'r') as f:
            corrected_reads = f['corrected_reads'][:]
            global_rates = f['global_rates'][:]
            basis_functions = f['basis_functions'][:]
            weights = f['weights'][:]

            return cls(
                corrected_reads=corrected_reads,
                global_rates=global_rates,
                basis_functions=basis_functions,
                weights=weights)


class PhenotypeSelectionResult:
    """
    Data model for the results of one job in phenotype selection k-fold cross validation
    """
    def __init__(self,
                 mask: np.ndarray,
                 cell_prob_trace: np.ndarray,
                 expression_trace: np.ndarray,
                 beta_trace: np.ndarray,
                 cell_num_trace: np.ndarray,
                 log_lh_train_trace: np.ndarray,
                 log_lh_test_trace: np.ndarray,
                 n_components: int,
                 lam: float,
                 fold_number: int):
        """
        :param mask: <N tissue spots> length boolean array, False if the spot is being held out, True otherwise
        :param cell_prob_trace: <N samples> x <N tissue spots> x <N components + 1> matrix
        :param expression_trace: <N samples> x <N components> x <N markers> matrix
        :param beta_trace: <N samples> x <N components> matrix
        :param cell_num_trace: <N samples> x <N tissue spots> x <N components + 1> matrix
        :param log_lh_train_trace: -log likelihood for training set
        :param log_lh_test_trace: -log likelihood for test set
        :param n_components: Number of cell types for posterior distribution being sampled in this job
        :param lam: Lambda parameter of posterior distribution for this job
        :param fold_number: Index into the k-fold series
        """
        self.fold_number = fold_number
        self.lam = lam
        self.n_components = n_components
        self.mask = mask
        self.log_lh_test_trace = log_lh_test_trace
        self.log_lh_train_trace = log_lh_train_trace
        self.cell_num_trace = cell_num_trace
        self.beta_trace = beta_trace
        self.expression_trace = expression_trace
        self.cell_prob_trace = cell_prob_trace

    def save(self, path):
        with h5py.File(path, 'w') as f:
            f['mask'] = self.mask
            f['log_lh_test_trace'] = self.log_lh_test_trace
            f['log_lh_train_trace'] = self.log_lh_train_trace
            f['cell_num_trace'] = self.cell_num_trace
            f['beta_trace'] = self.beta_trace
            f['expression_trace'] = self.expression_trace
            f['cell_prob_trace'] = self.cell_prob_trace
            f.attrs['fold_number'] = self.fold_number
            f.attrs['lam'] = self.lam
            f.attrs['n_components'] = self.n_components

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, 'r') as f:
            mask = f['mask'][:]
            log_lh_test_trace = f['log_lh_test_trace'][:]
            log_lh_train_trace = f['log_lh_train_trace'][:]
            cell_num_trace = f['cell_num_trace'][:]
            beta_trace = f['beta_trace'][:]
            expression_trace = f['expression_trace'][:]
            cell_prob_trace = f['cell_prob_trace'][:]
            fold_number = f.attrs['fold_number']
            lam = f.attrs['lam']
            n_components = f.attrs['n_components']

            return cls(
                mask=mask,
                log_lh_test_trace=log_lh_test_trace,
                log_lh_train_trace=log_lh_train_trace,
                cell_prob_trace=cell_prob_trace,
                expression_trace=expression_trace,
                beta_trace=beta_trace,
                cell_num_trace=cell_num_trace,
                fold_number=fold_number,
                lam=lam,
                n_components=n_components)


class DeconvolutionResult:
    """
    Data model for the results of sampling from the deconvolution posterior distribution.
    """
    def __init__(self,
                 cell_prob_trace: np.ndarray,
                 expression_trace: np.ndarray,
                 beta_trace: np.ndarray,
                 cell_num_trace: np.ndarray,
                 reads_trace: np.ndarray,
                 lam2: float,
                 n_components: int):
        """

        :param cell_prob_trace: <N samples> x <N tissue spots> x <N components + 1> matrix
        :param expression_trace: <N samples> x <N components> x <N markers> matrix
        :param beta_trace: <N samples> x <N components> matrix
        :param cell_num_trace: <N samples> x <N tissue spots> x <N components + 1> matrix
        :param reads_trace: <N samples> x <N tissue spots> x <N markers> x <N components>
        :param lam2: lambda smoothing parameter used for the posterior distribution
        :param n_components: N components value for the posterior distribution
        """
        self.reads_trace = reads_trace
        self.cell_prob_trace = cell_prob_trace
        self.expression_trace = expression_trace
        self.beta_trace = beta_trace
        self.cell_num_trace = cell_num_trace
        self.lam2 = lam2
        self.n_components = n_components

    def save(self, path):
        with h5py.File(path, 'w') as f:
            f['cell_prob_trace'] = self.cell_prob_trace
            f['expression_trace'] = self.expression_trace
            f['beta_trace'] = self.beta_trace
            f['cell_num_trace'] = self.cell_num_trace
            f['reads_trace'] = self.reads_trace
            f.attrs['lam2'] = self.lam2
            f.attrs['n_components'] = self.n_components

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, 'r') as f:
            cell_prob_trace = f['cell_prob_trace'][:]
            expression_trace = f['expression_trace'][:]
            beta_trace = f['beta_trace'][:]
            cell_num_trace = f['cell_num_trace'][:]
            reads_trace = f['reads_trace'][:]
            lam2 = f.attrs['lam2']
            n_components = f.attrs['n_components']

            return cls(
                cell_prob_trace=cell_prob_trace,
                expression_trace=expression_trace,
                beta_trace=beta_trace,
                cell_num_trace=cell_num_trace,
                reads_trace=reads_trace,
                lam2=lam2,
                n_components=n_components)


class SpatialDifferentialExpressionResult:
    """
    Data model for results from sampling from the spatial differential expression posterior distribution.
    """
    def __init__(self,
                 w_samples: np.ndarray,
                 c_samples: np.ndarray,
                 gamma_samples: np.ndarray,
                 h_samples: np.ndarray,
                 v_samples: np.ndarray,
                 theta_samples: np.ndarray):
        """
        :param w_samples: <N samples> x <N components> x <N spatial patterns + 1> x <N tissue spots>
        :param c_samples: <N samples> x <N markers> x <N components>
        :param gamma_samples: <N samples> x <N components> x <N spatial patterns + 1>
        :param h_samples: <N samples> x <N markers> x <N components>
        :param v_samples: <N samples> x <N markers> x <N components>
        :param theta_samples: <N samples> x <N tissue spots> x <N markers> x <N components>
        """
        self.theta_samples = theta_samples
        self.v_samples = v_samples
        self.h_samples = h_samples
        self.gamma_samples = gamma_samples
        self.c_samples = c_samples
        self.w_samples = w_samples

    def save(self, path):
        with h5py.File(path, 'w') as f:
            f['theta_samples'] = self.theta_samples
            f['v_samples'] = self.v_samples
            f['h_samples'] = self.h_samples
            f['gamma_samples'] = self.gamma_samples
            f['c_samples'] = self.c_samples
            f['w_samples'] = self.w_samples

    @property
    def n_spatial_patterns(self):
        return self.gamma_samples.shape[2] - 1

    @property
    def n_components(self):
        return self.c_samples.shape[2]

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, 'r') as f:
            theta_samples = f['theta_samples'][:]
            v_samples = f['v_samples'][:]
            h_samples = f['h_samples'][:]
            gamma_samples = f['gamma_samples'][:]
            c_samples = f['c_samples'][:]
            w_samples = f['w_samples'][:]

            return cls(
                theta_samples=theta_samples,
                v_samples=v_samples,
                h_samples=h_samples,
                gamma_samples=gamma_samples,
                c_samples=c_samples,
                w_samples=w_samples)
