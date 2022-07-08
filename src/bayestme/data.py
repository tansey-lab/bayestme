import numpy as np
import pandas as pd
import scipy.io as io
import os
import glob
import logging
import h5py

from enum import Enum
from typing import Iterable, Optional

from . import utils

logger = logging.getLogger(__name__)


class Layout(Enum):
    HEX = 1
    SQUARE = 2


class SpatialExpressionDataset:
    def __init__(self,
                 raw_counts: np.ndarray,
                 positions: Optional[np.ndarray],
                 tissue_mask: Optional[np.ndarray],
                 gene_names: np.ndarray,
                 layout: Layout):
        self.layout = layout
        self.gene_names = gene_names
        self.tissue_mask = tissue_mask
        self.positions = positions.astype(int)
        self.raw_counts = raw_counts
        self.positions_tissue = positions[:, tissue_mask].astype(int)
        self.edges = utils.get_edges(self.positions_tissue, layout=self.layout.value)

    @property
    def reads(self) -> np.ndarray:
        return self.raw_counts[self.tissue_mask]

    @property
    def n_spot_in(self) -> int:
        return self.raw_counts[self.tissue_mask].shape[0]

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
        Inputs:
            data_path:  /path/to/spaceranger/outs
                        should contain at least 1) /raw_feature_bc_matrix for raw count matrix
                                                2) /filtered_feature_bc_matrix for filtered count matrix
                                                3) /spatial for position list
            layout:     Visim(hex)  1
                        ST(square)  2
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
        Load data from tsv count matrix containing only in-tissue spots where the count matrix is a tsv file of shape G by N
        the column names and row names are position and gene names respectively
        Inputs:
            data_path:  /path/to/count_matrix
            layout:     Visim(hex)  1
                        ST(square)  2
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
    def __init__(self,
                 corrected_reads: np.ndarray,
                 global_rates: np.ndarray,
                 basis_functions: np.ndarray,
                 weights: np.ndarray):
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


class DeconvolutionResult:
    def __init__(self,
                 cell_prob_trace: np.ndarray,
                 expression_trace: np.ndarray,
                 beta_trace: np.ndarray,
                 cell_num_trace: np.ndarray,
                 reads_trace: np.ndarray,
                 lam2: float,
                 n_components: int):
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
    def __init__(self,
                 w_samples: np.ndarray,
                 c_samples: np.ndarray,
                 gamma_samples: np.ndarray,
                 h_samples: np.ndarray,
                 v_samples: np.ndarray,
                 theta_samples: np.ndarray):
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
