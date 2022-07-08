import numpy as np
import pandas as pd
import scipy.io as io
import os
import glob
import logging

from . import utils
from .model_bkg import GraphFusedMultinomial
from .bayestme_data import RawSTData, DeconvolvedSTData, SpatialExpression

logger = logging.getLogger(__name__)


class BayesTME:
    def __init__(self, exp_name='BayesTME', storage_path=None):
        # set up experiment name
        self.exp_name = exp_name
        # set storage path of all generated results
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(exp_name, '_results/')

    def load_data_from_spaceranger(self, data_path, layout=1):
        '''
        Load data from spaceranger /outputs folder
        Inputs:
            data_path:  /path/to/spaceranger/outs
                        should contain at least 1) /raw_feature_bc_matrix for raw count matrix
                                                2) /filtered_feature_bc_matrix for filtered count matrix
                                                3) /spatial for position list
            layout:     Visim(hex)  1
                        ST(square)  2
        '''
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
        return RawSTData(
            data_name=self.exp_name,
            raw_count=raw_count.T,
            positions=positions.T,
            tissue_mask=tissue_mask,
            gene_names=features,
            layout=layout,
            storage_path=self.storage_path)

    def load_data_from_count_mat(self, data_path, layout=2):
        '''
        Load data from tsv count matrix containing only in-tissue spots where the count matrix is a tsv file of shape G by N
        the column names and row names are position and gene names respectively
        Inputs:
            data_path:  /path/to/count_matrix
            layout:     Visim(hex)  1
                        ST(square)  2
        '''
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
        return RawSTData(
            data_name=self.exp_name,
            raw_count=count_mat,
            positions=positions,
            tissue_mask=tissue_mask,
            gene_names=features,
            layout=layout,
            storage_path=self.storage_path)

    def cleaning_data(self, stdata, n_top=50, max_steps=5):
        '''

        '''
        return stdata.bleeding_correction(n_top, max_steps)

    def kfold(self, stdata, cluster_storage, n_fold=5, n_splits=15, n_samples=100, n_burn=2000, n_thin=5, lda=0):
        '''
        Auto-tuning 1) number of cell-types         K
                    2) spatial smoothing parameter  lam
        '''
        return stdata.k_fold(cluster_storage, n_fold, n_splits, n_samples, n_burn, n_thin, lda)

    def deconvolve(self,
                   STData,
                   n_gene=None,
                   n_components=None,
                   lam2=None,
                   n_samples=100,
                   n_burnin=1000,
                   n_thin=10,
                   random_seed=0,
                   bkg=False,
                   lda=False,
                   cv=False,
                   max_ncell=120):
        '''

        Inputs:
            data:           either (1) RawSTData
                                or (2) CleanedSTData
            n_gene:         int or list
                            number or list of indices of the genes to look at
            n_componets:    int 
                            number of celltypes to segment (if known)
                            otherwise can be determined by cross validation
            lam2:           real positive number 
                            parameter controls the degree of spatial smoothing
                            recommend range (1e-2, 1e6) the less lam2 the more smoothing
                            otherwise can be determined by cross validation
            n_sample:       int
                            number of posterior samples
            n_burnin:       int
                            number of burn-in samples
            n_thin:         int
                            number of thinning
            random_seed:    int
                            random state
            bkg:            boolean
                            if fit with background noise
            lda:            boolean
                            if initialize model with LDA, converges faster but no garantee of correctness
                            recommend set to False
        '''
        # load position, and spatial layout
        self.pos = STData.positions_tissue
        self.layout = STData.layout
        if self.layout == 1:
            spatial = 'Visim(hex)'
        else:
            spatial = 'ST(square)'
        # generate edge graph from spot positions and ST layout
        self.edges = utils.get_edges(self.pos, self.layout)
        self.n_components = n_components
        self.lam2 = lam2
        # detetermine the number of spots
        self.n_nodes = STData.Reads.shape[0]

        # load the count matrix
        if n_gene is None:
            self.n_gene = STData.Reads.shape[1]
            Observation = STData.Reads
        elif isinstance(n_gene, (list, np.ndarray)):
            self.n_gene = len(n_gene)
            Observation = STData.Reads[:, n_gene]
        elif isinstance(n_gene, int):
            self.n_gene = min(n_gene, STData.Reads.shape[1])
            top = np.argsort(np.std(np.log(1 + STData.Reads), axis=0))[::-1]
            Observation = STData.Reads[:, top[:self.n_gene]]
        else:
            raise ValueError('n_gene must be a integer or a list of indices of genes')

        np.random.seed(random_seed)

        # initialize the model
        if n_components is not None:
            self.n_components = n_components
        else:
            raise Exception('use cv to determine number of cell_types')
        if lam2 is not None:
            self.lam2 = lam2
        else:
            raise Exception('use cv to determine spatial smoothing parameter')
        results_path = os.path.join(self.storage_path, 'results/')
        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        logger.info('experiment: {}, lambda {}, {} components'.format(self.exp_name, lam2, n_components))
        logger.info(
            '\t {} lda, {} layout, {} max cells, {}({}) gene'.format(lda, spatial, max_ncell, self.n_gene, n_gene))
        logger.info('sampling: {} burn_in, {} samples, {} thinning'.format(n_burnin, n_samples, n_thin))
        logger.info('storage: {}'.format(results_path))

        gfm = GraphFusedMultinomial(n_components=n_components, edges=self.edges, observations=Observation,
                                    n_gene=self.n_gene, lam_psi=self.lam2,
                                    background_noise=bkg, lda_initialization=lda)

        cell_prob_trace = np.zeros((n_samples, self.n_nodes, self.n_components + 1))
        cell_num_trace = np.zeros((n_samples, self.n_nodes, self.n_components + 1))
        expression_trace = np.zeros((n_samples, self.n_components, self.n_gene))
        beta_trace = np.zeros((n_samples, self.n_components))
        reads_trace = np.zeros((n_samples, self.n_nodes, self.n_gene, self.n_components))
        if cv:
            loglhtest_trace = np.zeros(n_samples)
        loglhtrain_trace = np.zeros(n_samples)
        total_samples = n_samples * n_thin + n_burnin
        for step in range(total_samples):
            logger.info(f'Step {step}/{total_samples} ...', end='\r')
            # perform Gibbs sampling
            gfm.sample(Observation)
            # save the trace of GFMM parameters
            if step >= n_burnin and (step - n_burnin) % n_thin == 0:
                idx = (step - n_burnin) // n_thin
                cell_prob_trace[idx] = gfm.probs
                expression_trace[idx] = gfm.phi
                beta_trace[idx] = gfm.beta
                cell_num_trace[idx] = gfm.cell_num
                reads_trace[idx] = gfm.reads

        logger.info(f'Step {step + 1}/{total_samples} finished!')
        np.save(os.path.join(results_path, 'reads_trace.npy').format(self.exp_name, self.n_components, self.lam2), reads_trace)

        return DeconvolvedSTData(stdata=STData, cell_prob_trace=cell_prob_trace, expression_trace=expression_trace,
                                 beta_trace=beta_trace, cell_num_trace=cell_num_trace, lam=self.lam2)

    def spatial_expression(self, DecovolvedData, n_spatial_patterns=10, n_samples=100, n_burn=100, n_thin=5, n_gene=50,
                           simple=False):
        return SpatialExpression(stdata=DecovolvedData, n_spatial_patterns=n_spatial_patterns, n_samples=n_samples,
                                 n_burn=n_burn, n_thin=n_thin, simple=simple, n_gene=n_gene)
