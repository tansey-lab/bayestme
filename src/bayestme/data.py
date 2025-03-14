import glob
import logging
import os
from typing import Optional, List

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.io as io
import scipy.sparse.csc
from scipy.sparse import issparse
from scanpy import read_10x_h5, read_10x_mtx
from scipy.sparse import csr_matrix

from bayestme import utils
from bayestme.common import ArrayType, Layout

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
RELATIVE_MEAN_EXPRESSION_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_relative_mean_expression"
OMEGA_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_omega"
RELATIVE_EXPRESSION_ATTR = f"{BAYESTME_ANNDATA_PREFIX}_relative_expression"
POSITIONS_X_COLUMN = "array_col"
POSITIONS_Y_COLUMN = "array_row"
REAL_POSITION_X_COLUMN = "pxl_col_in_fullres"
REAL_POSITION_Y_COLUMN = "pxl_row_in_fullres"
VISIUM_SPATIAL_COLUMNS = [
    "barcode",
    "in_tissue",
    POSITIONS_Y_COLUMN,
    POSITIONS_X_COLUMN,
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


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
    edges: np.ndarray,
    barcodes: Optional[np.array] = None,
):
    """
    Create an AnnData object from spatial expression data.

    :param counts: N x G read count matrix
    :param coordinates: N x 2 coordinate matrix
    :param tissue_mask: N length boolean array indicating in-tissue or out of tissue
    :param gene_names: N length string array of gene names
    :param layout: Layout enum
    :param edges: N x 2 array indicating adjacency between spots
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
    connectivities = csr_matrix(
        (np.array([True] * edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(adata.n_obs, adata.n_obs),
        dtype=bool,
    )
    adata.obsp[CONNECTIVITIES_ATTR] = connectivities

    return adata


def read_with_maybe_header(path, header):
    df = pd.read_csv(path)
    seen_header = df.columns
    if set(seen_header).issubset(set(header)):
        return df
    else:
        return pd.read_csv(path, names=header)


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
    def positions_tissue(self) -> ArrayType:
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
    def raw_counts(self) -> ArrayType:
        X = self.adata.X

        if issparse(X):
            return np.asarray(X.todense())
        else:
            return X

    @property
    def counts(self) -> ArrayType:
        X = self.adata[self.adata.obs[IN_TISSUE_ATTR]].X

        if issparse(X):
            return np.asarray(X.todense())
        else:
            return X

    @property
    def positions(self) -> ArrayType:
        return self.adata.obsm[SPATIAL_ATTR]

    @property
    def tissue_mask(self) -> ArrayType:
        return self.adata.obs[IN_TISSUE_ATTR].to_numpy()

    @property
    def gene_names(self) -> ArrayType:
        return self.adata.var_names

    @property
    def edges(self) -> ArrayType:
        return np.array(self.adata.obsp[CONNECTIVITIES_ATTR].nonzero()).T

    @property
    def layout(self) -> Layout:
        return Layout[self.adata.uns[LAYOUT_ATTR]]

    @property
    def n_cell_types(self) -> Optional[int]:
        if N_CELL_TYPES_ATTR in self.adata.uns:
            return self.adata.uns[N_CELL_TYPES_ATTR]

    @property
    def cell_type_probabilities(self) -> Optional[ArrayType]:
        if CELL_TYPE_PROB_ATTR in self.adata.obsm:
            return self.adata[self.tissue_mask].obsm[CELL_TYPE_PROB_ATTR]

    @property
    def cell_type_counts(self) -> Optional[ArrayType]:
        if CELL_TYPE_COUNT_ATTR in self.adata.obsm:
            return self.adata[self.tissue_mask].obsm[CELL_TYPE_COUNT_ATTR]

    @property
    def marker_gene_names(self) -> Optional[List[ArrayType]]:
        if MARKER_GENE_ATTR not in self.adata.varm:
            return

        outputs = []

        for i in range(self.n_cell_types):
            outputs.append(self.adata.var_names[self.marker_gene_indices[i]])

        return outputs

    @property
    def marker_gene_indices(self) -> Optional[List[ArrayType]]:
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
    def omega_difference(self) -> Optional[ArrayType]:
        if OMEGA_DIFFERENCE_ATTR not in self.adata.varm:
            return

        return self.adata.varm[OMEGA_DIFFERENCE_ATTR].T

    @property
    def relative_mean_expression(self) -> Optional[ArrayType]:
        if RELATIVE_MEAN_EXPRESSION_ATTR not in self.adata.varm:
            return

        return self.adata.varm[RELATIVE_MEAN_EXPRESSION_ATTR].T

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
        edges: np.ndarray,
        barcodes: Optional[np.array] = None,
    ):
        """
        Construct SpatialExpressionDataset directly from numpy arrays.

        :param raw_counts: An <N spots> x <N markers> matrix.
        :param positions: An <N spots> x 2 matrix of spot coordinates.
        :param tissue_mask: An <N spot> length array of booleans. True if spot is in tissue, False if not.
        :param gene_names: An <M markers> length array of gene names.
        :param layout: Layout.SQUARE of the spots are in a square grid layout, Layout.HEX if the spots are
        :param edges: An <N spots> x 2 matrix of edges between spots.
        :param barcodes: List of UMI barcodes
        in a hex grid layout.
        """
        adata = create_anndata_object(
            counts=raw_counts,
            coordinates=positions,
            tissue_mask=tissue_mask,
            gene_names=gene_names,
            layout=layout,
            edges=edges,
            barcodes=barcodes,
        )
        return cls(adata)

    @classmethod
    def read_spaceranger(cls, data_path):
        """
        Load data from spaceranger /outputs folder

        :param data_path: Directory containing at least
            1) /raw_feature_bc_matrix for raw count matrix
            2) /filtered_feature_bc_matrix for filtered count matrix
            3) /spatial for position list
        :return: SpatialExpressionDataset
        """
        spatial_dir = glob.glob(os.path.join(data_path, "*spatial"))
        if spatial_dir:
            spatial_dir = spatial_dir[0]
        else:
            raise RuntimeError("No spatial directory found in spaceranger directory")

        tissue_positions_v1_path = os.path.join(
            spatial_dir, "tissue_positions_list.csv"
        )
        tissue_positions_v2_path = os.path.join(spatial_dir, "tissue_positions.csv")

        if os.path.exists(tissue_positions_v1_path) and os.path.isfile(
            tissue_positions_v1_path
        ):
            logger.info(
                f"Reading V1 tissue positions list at {tissue_positions_v1_path}"
            )
            positions_df = read_with_maybe_header(
                tissue_positions_v1_path, VISIUM_SPATIAL_COLUMNS
            )
        elif os.path.exists(tissue_positions_v2_path) and os.path.isfile(
            tissue_positions_v2_path
        ):
            logger.info(
                f"Reading V2 tissue positions list at {tissue_positions_v2_path}"
            )
            positions_df = read_with_maybe_header(
                tissue_positions_v2_path, VISIUM_SPATIAL_COLUMNS
            )
        else:
            raise RuntimeError("No positions list found in spaceranger directory")

        if positions_df.columns.tolist() != VISIUM_SPATIAL_COLUMNS:
            raise RuntimeError("Tissue positions list has unexpected columns")

        # find any file under data_path named *raw_feature_bc_matrix.h5
        raw_h5_path = glob.glob(os.path.join(data_path, "*raw_feature_bc_matrix.h5"))
        if raw_h5_path:
            raw_h5_path = raw_h5_path[0]
        else:
            raw_h5_path = None

        raw_mtx_path = glob.glob(os.path.join(data_path, "*raw_feature_bc_matrix"))

        if raw_mtx_path:
            raw_mtx_path = raw_mtx_path[0]
        else:
            raw_mtx_path = None

        if raw_h5_path:
            ad = read_10x_h5(raw_h5_path, gex_only=False)
        elif raw_mtx_path:
            ad = read_10x_mtx(raw_mtx_path, gex_only=False)
        else:
            raise RuntimeError(
                "No raw count matrix found in spaceranger directory"
                f"expected {raw_h5_path} or {raw_mtx_path}"
            )

        if (
            "feature_types" in ad.var
            and "Antibody Capture" in ad.var.feature_types.to_list()
        ):
            ad.var["id"] = ad.var.index
            ad.var["id"][ad.var.feature_types == "Antibody Capture"] = ad.var["id"][
                ad.var.feature_types == "Antibody Capture"
            ].apply(lambda x: x + "_protein")
            ad.var_names = ad.var["id"]
            del ad.var["id"]

        ad.var_names_make_unique()

        positions_df = positions_df.set_index("barcode")

        tissue_mask = (
            positions_df.loc[ad.obs_names, IN_TISSUE_ATTR]
            .apply(int)
            .values.astype(bool)
        )
        positions = (
            positions_df.loc[ad.obs_names, [POSITIONS_X_COLUMN, POSITIONS_Y_COLUMN]]
            .map(int)
            .values
        )

        physical_positions = (
            positions_df.loc[
                ad.obs_names, [REAL_POSITION_X_COLUMN, REAL_POSITION_Y_COLUMN]
            ]
            .map(float)
            .values
        )

        edges = utils.get_edges(physical_positions[tissue_mask], Layout.HEX)

        gene_names = ad.var_names.values
        barcodes = ad.obs_names.values
        raw_count = ad.X

        return cls.from_arrays(
            raw_counts=raw_count,
            positions=positions,
            tissue_mask=tissue_mask,
            gene_names=gene_names,
            layout=Layout.HEX,
            edges=edges,
            barcodes=barcodes,
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
            edges=utils.get_edges(positions, Layout.SQUARE),
        )

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        return cls(anndata.read_h5ad(path))

    def copy(self) -> "SpatialExpressionDataset":
        """
        Return a copy of this object
        :return: A copy of this object
        """
        ad = self.adata.copy()
        return SpatialExpressionDataset(ad)


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
        :param log_lh_train_trace: total log likelihood for each sample calculated over the non-held out spots
        :param log_lh_test_trace: total log likelihood for each sample calculated over the held out spots
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
        cell_num_total_trace: np.ndarray,
        lam2: float,
        n_components: int,
        losses: Optional[np.ndarray] = None,
    ):
        """

        :param cell_prob_trace: <N samples> x <N tissue spots> x <N components> matrix
        :param expression_trace: <N samples> x <N components> x <N markers> matrix
        :param beta_trace: <N samples> x <N components> matrix
        :param cell_num_total_trace: <N samples> x <N tissue spots> matrix
        :param lam2: lambda smoothing parameter used for the posterior distribution
        :param n_components: N components value for the posterior distribution
        :param losses: Training loss (if applicable for inference method)
        """
        self.cell_prob_trace = cell_prob_trace
        self.expression_trace = expression_trace
        self.beta_trace = beta_trace
        self.cell_num_total_trace = cell_num_total_trace
        self.lam2 = lam2
        self.n_components = n_components
        self.losses = losses

    def save(self, path):
        with h5py.File(path, "w") as f:
            f["cell_prob_trace"] = self.cell_prob_trace
            f["expression_trace"] = self.expression_trace
            f["beta_trace"] = self.beta_trace
            f["cell_num_total_trace"] = self.cell_num_total_trace
            f["reads_trace"] = self.reads_trace
            if self.losses is not None:
                f["losses"] = self.losses
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

    @property
    def relative_mean_expression(self):
        """
        Return a matrix of average expression in this cell type, divided by the average expression in all cell types.

        :return: An <N cell types> x <N markers> floating point matrix.
        """
        expression = np.zeros(
            shape=self.expression_trace.shape[1:], dtype=np.float64
        )  # n_components, n_genes
        gene_expression = self.expression_trace.mean(axis=0)  # n_components, n_genes
        for k in range(self.n_components):
            mask = np.arange(self.n_components) != k
            out_group_mean = gene_expression[mask].mean(axis=0)  # n_genes
            expression[k] = gene_expression[k] / out_group_mean

        return expression

    @property
    def reads_trace(self):
        """
        :return: <N Samples> x <N Spots> x <N Genes> x <N Cell Types>
        """
        number_of_cells_per_component = (
            self.cell_prob_trace.T * self.cell_num_total_trace.T
        ).T * self.beta_trace[:, None, :]
        result = (
            number_of_cells_per_component[:, :, :, None]
            * self.expression_trace[:, None, :, :]
        )
        return np.transpose(result, (0, 1, 3, 2))

    @property
    def cell_num_trace(self):
        return (self.cell_num_total_trace.T * self.cell_prob_trace.T).T

    @property
    def nb_probs(self):
        rates = (
            self.cell_prob_trace[..., None]
            * (self.beta_trace[..., None] * self.expression_trace)[:, None, ...]
        )
        return rates.sum(axis=2) / rates.sum(axis=(2, 3))[..., None]

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
            cell_num_total_trace = f["cell_num_total_trace"][:]
            lam2 = f.attrs["lam2"]
            n_components = f.attrs["n_components"]

            if "losses" in f:
                losses = f["losses"][:]
            else:
                losses = None

            return cls(
                cell_prob_trace=cell_prob_trace,
                expression_trace=expression_trace,
                beta_trace=beta_trace,
                cell_num_total_trace=cell_num_total_trace,
                lam2=lam2,
                n_components=n_components,
                losses=losses,
            )


class SpatialDifferentialExpressionResult:
    """
    Data model for results from sampling from the spatial differential expression posterior distribution.
    """

    def __init__(self, w_hat: np.array, v_hat: np.array, losses: np.array):
        """
        :param w_hat: <N Cell Types> x <N Spatial Patterns> x <N tissue spots>
        :param v_hat: <N Cell Types> x <N Genes> x <N Spatial Patterns>
        :param losses: <N Iter> array of loss values
        """
        self.w_hat = w_hat
        self.v_hat = v_hat
        self.losses = losses

    def save(self, path):
        with h5py.File(path, "w") as f:
            f["w_hat"] = self.w_hat
            f["v_hat"] = self.v_hat
            f["losses"] = self.losses

    @property
    def n_spatial_patterns(self):
        return self.w_hat.shape[1]

    @property
    def n_components(self):
        return self.w_hat.shape[0]

    @property
    def spatial_hat(self):
        """
        Return the per spot gene expression learned by the factor model
        """
        return np.exp((self.w_hat[:, None] * self.v_hat[..., None, None]).sum(axis=2))

    @classmethod
    def read_h5(cls, path):
        """
        Read this class from an h5 archive
        :param path: Path to h5 file.
        :return: SpatialExpressionDataset
        """
        with h5py.File(path, "r") as f:
            w_hat = f["w_hat"][:]
            v_hat = f["v_hat"][:]
            losses = f["losses"][:]

            return cls(w_hat=w_hat, v_hat=v_hat, losses=losses)


def add_deconvolution_results_to_dataset(
    stdata: SpatialExpressionDataset, result: DeconvolutionResult
):
    """
    Modify stdata in-place to annotate it with selected marker genes

    :param stdata: data.SpatialExpressionDataset to modify
    :param result: data.DeconvolutionResult to use
    """
    cell_num_matrix = result.cell_num_trace.mean(axis=0)
    cell_prob_matrix = result.cell_prob_trace.mean(axis=0)

    cell_prob_matrix_full = np.zeros((stdata.n_spot, cell_prob_matrix.shape[1]))

    cell_prob_matrix_full[stdata.tissue_mask] = cell_prob_matrix

    cell_num_matrix_full = np.zeros((stdata.n_spot, cell_num_matrix.shape[1]))

    cell_num_matrix_full[stdata.tissue_mask] = cell_num_matrix

    stdata.adata.obsm[CELL_TYPE_PROB_ATTR] = cell_prob_matrix_full
    stdata.adata.obsm[CELL_TYPE_COUNT_ATTR] = cell_num_matrix_full

    stdata.adata.uns[N_CELL_TYPES_ATTR] = result.n_components
    stdata.adata.varm[OMEGA_DIFFERENCE_ATTR] = result.omega_difference.T
    stdata.adata.varm[OMEGA_ATTR] = result.omega.T
    stdata.adata.varm[RELATIVE_EXPRESSION_ATTR] = result.relative_expression.T
    stdata.adata.varm[RELATIVE_MEAN_EXPRESSION_ATTR] = result.relative_mean_expression.T
