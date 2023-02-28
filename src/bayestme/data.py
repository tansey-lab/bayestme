import glob
import logging
import os
from enum import Enum
from typing import Optional, List

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.io as io
import scipy.sparse.csc
from scipy.sparse import csr_matrix

from . import utils

logger = logging.getLogger(__name__)

IN_TISSUE_ATTR = "in_tissue"
SPATIAL_ATTR = "spatial"
LAYOUT_ATTR = "layout"
CONNECTIVITIES_ATTR = "connectivities"
BAYESTME_ANNDATA_PREFIX = "bayestme"
N_CELL_TYPES_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_n_cell_types"
CELL_TYPE_COUNT_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_cell_type_counts"
CELL_TYPE_PROB_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_cell_type_probabilities"
MARKER_GENE_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_cell_type_marker"
OMEGA_DIFFERENCE_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_omega_difference"


class Layout(Enum):
    HEX = 1
    SQUARE = 2


def is_csv(fn: str):
    return fn.lower().endswith("csv.gz") or fn.lower().endswith("csv")


def is_tsv(fn: str):
    return fn.lower().endswith("tsv.gz") or fn.lower().endswith("tsv")


def is_csv_tsv(fn: str):
    return is_csv(fn) or is_tsv(fn)


def create_anndata_object(
    counts: np.ndarray,
    coordinates: Optional[np.ndarray],
    tissue_mask: Optional[np.ndarray],
    gene_names: np.ndarray,
    layout: Layout,
    barcodes: Optional[np.array] = None,
):
    """
    Create an AnnData object from spatial expression data.

    :param counts: N x G read count matrix
    :param coordinates: N x 2 coordinate matrix
    :param tissue_mask: N length boolean array indicating in-tissue or out of tissue
    :param gene_names: N length string array of gene names
    :param layout: Layout enum
    :param barcodes: List of UMI barcodes
    :return: AnnData object containing all information provided.
    """
    coordinates = coordinates.astype(int)
    adata = anndata.AnnData(counts, obsm={SPATIAL_ATTR: coordinates})
    adata.obs[IN_TISSUE_ATTR] = tissue_mask
    adata.uns[LAYOUT_ATTR] = layout.name
    adata.var_names = gene_names
    if barcodes is not None:
        adata.obs_names = barcodes
    edges = utils.get_edges(coordinates[tissue_mask], layout.value)
    connectivities = csr_matrix(
        (np.array([True] * edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(adata.n_obs, adata.n_obs),
        dtype=np.bool,
    )
    adata.obsp[CONNECTIVITIES_ATTR] = connectivities

    return adata


class SpatialExpressionDataset:
    """
    Data model for holding read counts, their associated position information,
    and whether they come from tissue or non tissue spots.
    Also holds the names of the gene markers in the dataset.
    """

    def __init__(self, adata: anndata.AnnData):
        """
        :param adata: AnnData object
        """
        self.adata: anndata.AnnData = adata

    @property
    def reads(self) -> np.ndarray:
        return self.adata[self.adata.obs[IN_TISSUE_ATTR]].X

    @property
    def positions_tissue(self) -> np.ndarray:
        return self.adata[self.adata.obs[IN_TISSUE_ATTR]].obsm[SPATIAL_ATTR]

    @property
    def n_spot_in(self) -> int:
        return self.adata.obs[IN_TISSUE_ATTR].sum()

    @property
    def n_spot(self) -> int:
        return self.adata.n_obs

    @property
    def n_gene(self) -> int:
        return self.adata.n_vars

    @property
    def raw_counts(self) -> np.ndarray:
        return self.adata.X

    @property
    def positions(self) -> np.ndarray:
        return self.adata.obsm[SPATIAL_ATTR]

    @property
    def tissue_mask(self) -> np.array:
        return self.adata.obs[IN_TISSUE_ATTR].to_numpy()

    @property
    def gene_names(self) -> np.array:
        return self.adata.var_names

    @property
    def edges(self) -> np.ndarray:
        return np.array(self.adata.obsp[CONNECTIVITIES_ATTR].nonzero()).T

    @property
    def layout(self) -> Layout:
        return Layout[self.adata.uns[LAYOUT_ATTR]]

    @property
    def n_cell_types(self) -> Optional[int]:
        if N_CELL_TYPES_ATTR in self.adata.uns:
            return self.adata.uns[N_CELL_TYPES_ATTR]

    @property
    def cell_type_probabilities(self) -> Optional[np.ndarray]:
        if CELL_TYPE_PROB_ATTR in self.adata.obsm:
            return self.adata[self.tissue_mask].obsm[CELL_TYPE_PROB_ATTR]

    @property
    def cell_type_counts(self) -> Optional[np.ndarray]:
        if CELL_TYPE_COUNT_ATTR in self.adata.obsm:
            return self.adata[self.tissue_mask].obsm[CELL_TYPE_COUNT_ATTR]

    @property
    def marker_gene_names(self) -> Optional[List[np.ndarray]]:
        if MARKER_GENE_ATTR not in self.adata.varm:
            return

        outputs = []

        for i in range(self.n_cell_types):
            outputs.append(self.adata.var_names[self.marker_gene_indices[i]])

        return outputs

    @property
    def marker_gene_indices(self) -> Optional[List[np.ndarray]]:
        if MARKER_GENE_ATTR not in self.adata.varm:
            return
        outputs = []

        for i in range(self.n_cell_types):
            marker_gene_indices = self.adata.varm[MARKER_GENE_ATTR].T[i] >= 0
            marker_gene_order = self.adata.varm[MARKER_GENE_ATTR].T[i][
                marker_gene_indices
            ]

            outputs.append(
                np.arange(self.adata.n_vars)[marker_gene_indices][marker_gene_order]
            )

        return outputs

    @property
    def omega_difference(self) -> Optional[np.ndarray]:
        if OMEGA_DIFFERENCE_ATTR not in self.adata.varm:
            return

        return self.adata.varm[OMEGA_DIFFERENCE_ATTR].T

    def save(self, path):
        self.adata.write_h5ad(path)

    @classmethod
    def from_arrays(
        cls,
        raw_counts: np.ndarray,
        positions: Optional[np.ndarray],
        tissue_mask: Optional[np.ndarray],
        gene_names: np.ndarray,
        layout: Layout,
        barcodes: Optional[np.array] = None,
    ):
        """
        Construct SpatialExpressionDataset directly from numpy arrays.

        :param raw_counts: An <N spots> x <N markers> matrix.
        :param positions: An <N spots> x 2 matrix of spot coordinates.
        :param tissue_mask: An <N spot> length array of booleans. True if spot is in tissue, False if not.
        :param gene_names: An <M markers> length array of gene names.
        :param layout: Layout.SQUARE of the spots are in a square grid layout, Layout.HEX if the spots are
        :param barcodes: List of UMI barcodes
        in a hex grid layout.
        """
        adata = create_anndata_object(
            counts=raw_counts,
            coordinates=positions,
            tissue_mask=tissue_mask,
            gene_names=gene_names,
            layout=layout,
            barcodes=barcodes,
        )
        return cls(adata)

    @classmethod
    def read_legacy_h5(cls, path):
        with h5py.File(path, "r") as f:
            raw_counts = f["raw_counts"][:]
            positions = f["positions"][:]
            tissue_mask = f["tissue_mask"][:]
            gene_names = np.array([x.decode("utf-8") for x in f["gene_names"][:]])
            layout_name = f.attrs["layout"]
            layout = Layout[layout_name]

            return cls.from_arrays(
                raw_counts=raw_counts,
                positions=positions,
                tissue_mask=tissue_mask,
                gene_names=gene_names,
                layout=layout,
            )

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
        raw_count_path = os.path.join(data_path, "raw_feature_bc_matrix/matrix.mtx.gz")
        filtered_count_path = os.path.join(
            data_path, "filtered_feature_bc_matrix/matrix.mtx.gz"
        )
        features_path = os.path.join(data_path, "raw_feature_bc_matrix/features.tsv.gz")
        barcodes_path = os.path.join(data_path, "raw_feature_bc_matrix/barcodes.tsv.gz")

        tissue_positions_lists = glob.glob(
            os.path.join(data_path, "spatial/tissue_positions_list.*")
        )

        positions_path = [fn for fn in tissue_positions_lists if is_csv_tsv(fn)][0]

        if is_tsv(positions_path):
            positions_list = pd.read_csv(
                positions_path, sep="\t", header=None, index_col=0, names=None
            )
        elif is_csv(positions_path):
            positions_list = pd.read_csv(
                positions_path, header=None, index_col=0, names=None
            )
        else:
            raise RuntimeError("No positions list found in spaceranger directory")

        raw_count = np.array(io.mmread(raw_count_path).todense())
        filtered_count = np.array(io.mmread(filtered_count_path).todense())
        features = np.array(pd.read_csv(features_path, header=None, sep="\t"))[
            :, 1
        ].astype(str)
        barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")
        n_spots = raw_count.shape[1]
        n_genes = raw_count.shape[0]
        logger.info("detected {} spots, {} genes".format(n_spots, n_genes))
        pos = np.zeros((n_spots, 3))
        for i in range(n_spots):
            pos[i] = np.array(positions_list.loc[barcodes[0][i]][:3])
        tissue_mask = pos[:, 0] == 1
        positions = pos[:, 1:]
        n_spot_in = tissue_mask.sum()
        logger.info("\t {} spots in tissue sample".format(n_spot_in))
        all_counts = raw_count.sum()
        tissue_counts = filtered_count.sum()
        logger.info(
            "\t {:.3f}% UMI counts bleeds out".format(
                (1 - tissue_counts / all_counts) * 100
            )
        )

        return cls.from_arrays(
            raw_counts=raw_count.T,
            positions=positions,
            tissue_mask=tissue_mask,
            gene_names=features,
            layout=layout,
            barcodes=barcodes[0].to_numpy(),
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
        raw_data = pd.read_csv(data_path, sep="\t")
        count_mat = raw_data.values[:, 1:].T.astype(int)
        features = np.array([x.split(" ")[0] for x in raw_data.values[:, 0]])
        n_spots = count_mat.shape[0]
        n_genes = count_mat.shape[1]
        logger.info("detected {} spots, {} genes".format(n_spots, n_genes))
        positions = np.zeros((n_spots, 2))
        for i in range(n_spots):
            spot_pos = raw_data.columns[1:][i].split("x")
            positions[i, 0] = int(spot_pos[0])
            positions[i, 1] = int(spot_pos[1])
        positions = positions.astype(int)
        tissue_mask = np.ones(n_spots).astype(bool)

        return cls.from_arrays(
            raw_counts=count_mat,
            positions=positions,
            tissue_mask=tissue_mask,
            gene_names=features,
            layout=layout,
        )

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        return cls(anndata.read_h5ad(path))


class BleedCorrectionResult:
    """
    Data model for the results of bleeding correction.
    """

    def __init__(
        self,
        corrected_reads: np.ndarray,
        global_rates: np.ndarray,
        basis_functions: np.ndarray,
        weights: np.ndarray,
    ):
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
        with h5py.File(path, "w") as f:
            f["corrected_reads"] = self.corrected_reads
            f["global_rates"] = self.global_rates
            f["basis_functions"] = self.basis_functions
            f["weights"] = self.weights

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, "r") as f:
            corrected_reads = f["corrected_reads"][:]
            global_rates = f["global_rates"][:]
            basis_functions = f["basis_functions"][:]
            weights = f["weights"][:]

            return cls(
                corrected_reads=corrected_reads,
                global_rates=global_rates,
                basis_functions=basis_functions,
                weights=weights,
            )


class PhenotypeSelectionResult:
    """
    Data model for the results of one job in phenotype selection k-fold cross validation
    """

    def __init__(
        self,
        mask: np.ndarray,
        cell_prob_trace: np.ndarray,
        expression_trace: np.ndarray,
        beta_trace: np.ndarray,
        cell_num_trace: np.ndarray,
        log_lh_train_trace: np.ndarray,
        log_lh_test_trace: np.ndarray,
        n_components: int,
        lam: float,
        fold_number: int,
    ):
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
        with h5py.File(path, "w") as f:
            f["mask"] = self.mask
            f["log_lh_test_trace"] = self.log_lh_test_trace
            f["log_lh_train_trace"] = self.log_lh_train_trace
            f["cell_num_trace"] = self.cell_num_trace
            f["beta_trace"] = self.beta_trace
            f["expression_trace"] = self.expression_trace
            f["cell_prob_trace"] = self.cell_prob_trace
            f.attrs["fold_number"] = self.fold_number
            f.attrs["lam"] = self.lam
            f.attrs["n_components"] = self.n_components

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, "r") as f:
            mask = f["mask"][:]
            log_lh_test_trace = f["log_lh_test_trace"][:]
            log_lh_train_trace = f["log_lh_train_trace"][:]
            cell_num_trace = f["cell_num_trace"][:]
            beta_trace = f["beta_trace"][:]
            expression_trace = f["expression_trace"][:]
            cell_prob_trace = f["cell_prob_trace"][:]
            fold_number = f.attrs["fold_number"]
            lam = f.attrs["lam"]
            n_components = f.attrs["n_components"]

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
                n_components=n_components,
            )


class DeconvolutionResult:
    """
    Data model for the results of sampling from the deconvolution posterior distribution.
    """

    def __init__(
        self,
        cell_prob_trace: np.ndarray,
        expression_trace: np.ndarray,
        beta_trace: np.ndarray,
        cell_num_trace: np.ndarray,
        reads_trace: np.ndarray,
        lam2: float,
        n_components: int,
    ):
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
        with h5py.File(path, "w") as f:
            f["cell_prob_trace"] = self.cell_prob_trace
            f["expression_trace"] = self.expression_trace
            f["beta_trace"] = self.beta_trace
            f["cell_num_trace"] = self.cell_num_trace
            f["reads_trace"] = self.reads_trace
            f.attrs["lam2"] = self.lam2
            f.attrs["n_components"] = self.n_components

    def align_celltype(self, sc_expression, n=50):
        """
        reorder the deconvolution results, aligned the detected cell type with the given scrna rerference
        :param sc_expression: K by G matrix, K cell types and G genes
        :returen: the ordering of the deconvolution result that matches the given scref
        """
        from scipy.stats import pearsonr

        expression_post = self.expression_trace[:].mean(axis=0)
        celltype_order = np.zeros(self.n_components)
        for ct_idx in range(self.n_components):
            ct_filter = np.zeros(self.n_components).astype(bool)
            ct_filter[ct_idx] = True
            score = (
                sc_expression[ct_filter][0] - sc_expression[~ct_filter].max(axis=0)
            ) / np.clip(sc_expression[ct_filter][0], 1e-6, None)
            n_marker = int(min((score > 0.1).sum(), n))
            gene_idx = score.argsort()[::-1][:n_marker]
            score = np.zeros(self.n_components)
            for i in range(self.n_components):
                score[i] = pearsonr(
                    sc_expression[ct_idx, gene_idx], expression_post[i, gene_idx]
                )[0]
            celltype_order[ct_idx] = score.argmax()
        return celltype_order.astype(int)

    @property
    def omega(self):
        """
        Return a matrix of ω_kg from equation 6 of the preprint

        :return: An <N cell types> x <N markers> floating point matrix.
        """
        omega = np.zeros(shape=self.expression_trace.shape[1:], dtype=np.float64)
        max_exp = self.expression_trace.max(axis=1)
        for k in range(self.n_components):
            omega[k] = (self.expression_trace[:, k, :] == max_exp).mean(axis=0)

        return omega

    @property
    def omega_difference(self):
        """
        Return a matrix of average ratio of expression/ maximum expression
        for each marker in each component

        This statistic represents the "overexpression" of a gene in a cell type, and is
        used for scaling the dot size in our marker gene plot.

        :return: An <N cell types> x <N markers> floating point matrix.
        """
        difference = np.zeros(shape=self.expression_trace.shape[1:], dtype=np.float64)
        max_exp = self.expression_trace.max(axis=1)
        for k in range(self.n_components):
            difference[k] = (self.expression_trace[:, k] / max_exp).mean(axis=0)

        return difference

    @property
    def relative_expression(self):
        """
        Return a matrix of average expression in this cell type, minus the max expression in all other cell types,
        divided by the maximum expression in all cell types. A higher number for this statistic represents a better
        candidate marker gene.

        This statistic is used as a tiebreaker criteria for marker gene selection when omega_kg values are
        equal.

        :return: An <N cell types> x <N markers> floating point matrix.
        """
        expression = np.zeros(shape=self.expression_trace.shape[1:], dtype=np.float64)
        gene_expression = self.expression_trace.mean(axis=0)
        for k in range(self.n_components):
            mask = np.arange(self.n_components) != k
            max_exp_g_k_prime = gene_expression[mask].max(axis=0)
            expression[k] = (gene_expression[k] - max_exp_g_k_prime) / np.max(
                gene_expression, axis=0
            )

        return expression

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, "r") as f:
            cell_prob_trace = f["cell_prob_trace"][:]
            expression_trace = f["expression_trace"][:]
            beta_trace = f["beta_trace"][:]
            cell_num_trace = f["cell_num_trace"][:]
            reads_trace = f["reads_trace"][:]
            lam2 = f.attrs["lam2"]
            n_components = f.attrs["n_components"]

            return cls(
                cell_prob_trace=cell_prob_trace,
                expression_trace=expression_trace,
                beta_trace=beta_trace,
                cell_num_trace=cell_num_trace,
                reads_trace=reads_trace,
                lam2=lam2,
                n_components=n_components,
            )


class SpatialDifferentialExpressionSamplerState:
    """
    Data model for internal SDE gibbs sampler state
    """

    def __init__(
        self,
        pickled_bit_generator: bytes,
        n_cell_types: int,
        n_nodes: int,
        n_signals: int,
        n_spatial_patterns: int,
        lam2: float,
        edges: np.ndarray,
        alpha: np.ndarray,
        W: np.ndarray,
        Gamma: np.ndarray,
        H: np.ndarray,
        C: np.ndarray,
        V: np.ndarray,
        Theta: np.ndarray,
        Omegas: np.ndarray,
        prior_vars: np.ndarray,
        Delta: scipy.sparse.csc.csc_matrix,
        DeltaT: scipy.sparse.csc.csc_matrix,
        Tau2: np.ndarray,
        Tau2_a: np.ndarray,
        Tau2_b: np.ndarray,
        Tau2_c: np.ndarray,
        Sigma0_inv: scipy.sparse.csc.csc_matrix,
        Cov_mats: np.ndarray,
    ):
        self.pickled_bit_generator = pickled_bit_generator
        self.n_cell_types = n_cell_types
        self.n_nodes = n_nodes
        self.n_signals = n_signals
        self.n_spatial_patterns = n_spatial_patterns
        self.lam2 = lam2
        self.edges = edges
        self.alpha = alpha
        self.W = W
        self.Gamma = Gamma
        self.H = H
        self.C = C
        self.V = V
        self.Theta = Theta
        self.Omegas = Omegas
        self.prior_vars = prior_vars
        self.Delta = Delta
        self.DeltaT = DeltaT
        self.Tau2 = Tau2
        self.Tau2_a = Tau2_a
        self.Tau2_b = Tau2_b
        self.Tau2_c = Tau2_c
        self.Sigma0_inv = Sigma0_inv
        self.Cov_mats = Cov_mats

    def save(self, path):
        with h5py.File(path, "w") as f:
            f.attrs["pickled_bit_generator"] = np.void(self.pickled_bit_generator)
            f.attrs["n_cell_types"] = self.n_cell_types
            f.attrs["n_nodes"] = self.n_nodes
            f.attrs["n_signals"] = self.n_signals
            f.attrs["n_spatial_patterns"] = self.n_spatial_patterns
            f.attrs["lam2"] = self.lam2
            f["edges"] = self.edges
            f["alpha"] = self.alpha
            f["W"] = self.W
            f["Gamma"] = self.Gamma
            f["H"] = self.H
            f["C"] = self.C
            f["V"] = self.V
            f["Theta"] = self.Theta
            f["Omegas"] = self.Omegas
            f["prior_vars"] = self.prior_vars
            f["Tau2"] = self.Tau2
            f["Tau2_a"] = self.Tau2_a
            f["Tau2_b"] = self.Tau2_b
            f["Tau2_c"] = self.Tau2_c
            f["Cov_mats"] = self.Cov_mats
            anndata._io.h5ad.write_sparse_as_dense(f, "Delta", self.Delta)
            anndata._io.h5ad.write_sparse_as_dense(f, "DeltaT", self.DeltaT)
            anndata._io.h5ad.write_sparse_as_dense(f, "Sigma0_inv", self.Sigma0_inv)

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, "r") as f:
            pickled_bit_generator = f.attrs["pickled_bit_generator"].tobytes()
            n_cell_types = f.attrs["n_cell_types"]
            n_nodes = f.attrs["n_nodes"]
            n_signals = f.attrs["n_signals"]
            n_spatial_patterns = f.attrs["n_spatial_patterns"]
            lam2 = f.attrs["lam2"]
            edges = f["edges"][:]
            alpha = f["alpha"][:]
            W = f["W"][:]
            Gamma = f["Gamma"][:]
            H = f["H"][:]
            C = f["C"][:]
            V = f["V"][:]
            Theta = f["Theta"][:]
            Omegas = f["Omegas"][:]
            prior_vars = f["prior_vars"][:]
            Tau2 = f["Tau2"][:]
            Tau2_a = f["Tau2_a"][:]
            Tau2_b = f["Tau2_b"][:]
            Tau2_c = f["Tau2_c"][:]
            Cov_mats = f["Cov_mats"][:]
            Delta = anndata._io.h5ad.read_dense_as_csc(f["Delta"])
            DeltaT = anndata._io.h5ad.read_dense_as_csc(f["DeltaT"])
            Sigma0_inv = anndata._io.h5ad.read_dense_as_csc(f["Sigma0_inv"])

            return cls(
                pickled_bit_generator=pickled_bit_generator,
                n_cell_types=n_cell_types,
                n_nodes=n_nodes,
                n_signals=n_signals,
                n_spatial_patterns=n_spatial_patterns,
                lam2=lam2,
                edges=edges,
                alpha=alpha,
                W=W,
                Gamma=Gamma,
                H=H,
                C=C,
                V=V,
                Theta=Theta,
                Omegas=Omegas,
                prior_vars=prior_vars,
                Tau2=Tau2,
                Tau2_a=Tau2_a,
                Tau2_b=Tau2_b,
                Tau2_c=Tau2_c,
                Cov_mats=Cov_mats,
                Delta=Delta,
                DeltaT=DeltaT,
                Sigma0_inv=Sigma0_inv,
            )


class SpatialDifferentialExpressionResult:
    """
    Data model for results from sampling from the spatial differential expression posterior distribution.
    """

    def __init__(
        self,
        w_samples: np.ndarray,
        c_samples: np.ndarray,
        gamma_samples: np.ndarray,
        h_samples: np.ndarray,
        v_samples: np.ndarray,
        theta_samples: np.ndarray,
    ):
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
        with h5py.File(path, "w") as f:
            f["theta_samples"] = self.theta_samples
            f["v_samples"] = self.v_samples
            f["h_samples"] = self.h_samples
            f["gamma_samples"] = self.gamma_samples
            f["c_samples"] = self.c_samples
            f["w_samples"] = self.w_samples

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
        with h5py.File(path, "r") as f:
            theta_samples = f["theta_samples"][:]
            v_samples = f["v_samples"][:]
            h_samples = f["h_samples"][:]
            gamma_samples = f["gamma_samples"][:]
            c_samples = f["c_samples"][:]
            w_samples = f["w_samples"][:]

            return cls(
                theta_samples=theta_samples,
                v_samples=v_samples,
                h_samples=h_samples,
                gamma_samples=gamma_samples,
                c_samples=c_samples,
                w_samples=w_samples,
            )
