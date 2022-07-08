import numpy as np
import matplotlib.pyplot as plt
import re
import os
import logging
import configparser
import math
import pathlib

from sklearn.model_selection import KFold
from matplotlib import colors
from scipy.stats import mode
from scipy.stats import pearsonr

from . import utils
from . import bleeding_correction as bleed
from . import bayestme_plot as bp
from . import communities
from .spatial_expression import SpatialDifferentialExpression

logger = logging.getLogger(__name__)


class RawSTData:
    def __init__(self, data_name, load=None, raw_count=None, positions=None, tissue_mask=None, gene_names=None,
                 layout=None, storage_path='./',
                 x_y_swap=False, invert=[0, 0], **kwargs):
        '''
        Inputs:
            load        /path/to/stored/data, if want to load from stored data
            raw_count   gene counts matrix of all spots in the spatial transcriptomics sample
                        (including spots outside tissue if possible)
            position    spatial coordinated of all spots in the spatial transcriptomics sample
                        (including spots outside tissue if possible)
            tissue_mask mask of in-tissue spots
            gene_names  gene names of sequenced genes in the spatial transcriptomics sample
            layout      Visim(hex)  1
                        ST(square)  2 
        '''
        if load:
            self.data_name = data_name
            self.load(load)
            self.n_spot_in = self.Reads.shape[0]
            self.n_gene = self.Reads.shape[1]
        else:
            self.data_name = data_name
            # clean up storage path
            if storage_path[-1] != '/':
                storage_path += '/'
            self.storage_path = storage_path
            if not os.path.isdir(storage_path):
                os.mkdir(storage_path)

            # store raw_count and position
            np.save(os.path.join(storage_path, 'raw_count.npy'), raw_count)
            np.save(os.path.join(storage_path, 'all_spots_position.npy'), positions)

            # set up basic parameters
            self.tissue_mask = tissue_mask
            self.gene_names = gene_names
            self.layout = layout
            self.storage_path = storage_path

            # get gene reads and spatial coordinates of in-tissue spots
            self.Reads = raw_count[tissue_mask]
            self.positions_tissue = positions[:, tissue_mask].astype(int)
            self.edges = utils.get_edges(self.positions_tissue, layout=self.layout)

            # set up plotting parameter
            self.x_y_swap = x_y_swap
            self.invert = invert

            self.n_spot_in = self.Reads.shape[0]
            self.n_gene = self.Reads.shape[1]
            self.filtering = np.zeros(self.n_gene).astype(bool)
            self.filter_genes = np.array([])
            self.selected_gene_idx = np.arange(self.n_gene)
            self.save()

    def set_plot_param(self, x_y_swap=False, invert=[0, 0], save=True):
        self.x_y_swap = x_y_swap
        self.invert = invert
        if save:
            self.save()

    def filter(self, n_gene=None, filter_type='ribo', pattern=None, filter_idx=None, spot_threshold=0.95, verbose=False,
               save=True):
        '''
        data preprocessing
        1.  narrow down number of genes to look at for cell-typing
            select top N gene by the standard deviation across spots
        2.  filter out confounding genes
            built-in filters:
            1)  'spots': universial genes that are observed in more than n% percent of the sample (defualt 95%)
            2)  'ribosome': ribosome genes, i.e. rpl and rps genes
            user can also pass in custom pattern or select gene idx for filtering
        '''
        # order genes by the standard deviation across spots
        top = np.argsort(np.std(np.log(1 + self.Reads), axis=0))[::-1]
        # apply n_gene filter
        if n_gene:
            n_gene_filter = min(n_gene, self.n_gene)
            logger.info('filtering top {} genes from original {} genes...'.format(n_gene_filter, self.n_gene))
            n_gene_filter = top[:n_gene_filter]
            self.Reads = self.Reads[:, n_gene_filter]
            self.gene_names = self.gene_names[n_gene_filter]
        else:
            n_gene_filter = top
            self.Reads = self.Reads[:, n_gene_filter]
            self.gene_names = self.gene_names[n_gene_filter]

        # define confounding genes filter
        if filter_type == 'spots':
            # built-in spots filter
            self.filtering = (self.Reads > 0).sum(axis=0) >= int(self.Reads.shape[0] * spot_threshold)
            logger.info('filtering out genes observed in {}% spots'.format(spot_threshold * 100))
        elif filter_type == 'ribosome':
            # built_in ribosome filter
            pattern = '[Rr][Pp][SsLl]'
            self.filtering = np.array([bool(re.match(pattern, g)) for g in self.gene_names])
        else:
            filter_type = filter_type if filter_type else 'custom'
            if pattern:
                # user-defined pattern filter
                self.filtering = np.array([bool(re.match(pattern, g)) for g in self.gene_names])
            elif filter_idx:
                # user defined gene idx filter
                self.filtering = np.zeros(self.n_gene).astype(bool)
                self.filtering[filter_idx] = True
            else:
                self.selected_gene_idx = n_gene_filter
                if save:
                    self.save()
                return
        logger.info('filtering out {} genes...'.format(filter_type))

        # apply confounding genes filter
        self.Reads = self.Reads[:, ~self.filtering]
        filtered_genes = self.gene_names[self.filtering]
        self.gene_names = self.gene_names[~self.filtering]
        self.selected_gene_idx = n_gene_filter[~self.filtering]
        self.n_spot_in = self.Reads.shape[0]
        self.n_gene = self.Reads.shape[1]
        logger.info('\t {} genes filtered out'.format(self.filtering.sum()))
        if verbose:
            logger.info(filtered_genes)
        np.save(self.storage_path + 'filtered_genes', filtered_genes)
        logger.info('Resulting dataset: {} spots, {} genes'.format(self.n_spot_in, self.n_gene))
        if save:
            self.save()

    def save(self):
        logger.info('Data saved in {}'.format(self.storage_path))
        np.save(
            os.path.join(self.storage_path, 'tissue_mask'),
            self.tissue_mask)
        np.save(
            os.path.join(self.storage_path, 'gene_names'),
            self.gene_names)
        np.save(
            os.path.join(self.storage_path, 'Reads'),
            self.Reads)
        np.save(
            os.path.join(self.storage_path, 'edges'),
            self.edges)
        params = np.array([self.layout, self.x_y_swap, self.invert[0], self.invert[1]])
        np.save(
            os.path.join(self.storage_path, 'param'),
            params)
        np.save(
            os.path.join(self.storage_path, 'filtering'),
            self.filtering)
        np.save(
            os.path.join(self.storage_path, 'filter_genes'),
            self.filter_genes)
        np.save(
            os.path.join(self.storage_path, 'selected_gene_idx'),
            self.selected_gene_idx)

    def load(self, load_path, storage_path=None):
        logger.info('Loading data from {}'.format(load_path))
        if not storage_path:
            storage_path = load_path
        else:
            self.storage_path = storage_path
        self.storage_path = storage_path
        # loading data
        raw_count = np.load(os.path.join(load_path, 'raw_count.npy'))
        positions = np.load(os.path.join(load_path, 'all_spots_position.npy'))
        tissue_mask = np.load(os.path.join(load_path, 'tissue_mask.npy'))
        gene_names = np.load(os.path.join(load_path, 'gene_names.npy'), allow_pickle=True)
        param = np.load(os.path.join(load_path, 'param.npy'))

        self.Reads = np.load(os.path.join(load_path, 'Reads.npy'))
        self.positions_tissue = positions[:, tissue_mask].astype(int)
        self.edges = np.load(os.path.join(load_path, 'edges.npy'))
        # set up other parameters
        self.tissue_mask = tissue_mask
        self.gene_names = gene_names
        self.layout = param[0]

        # set up plotting parameter
        self.x_y_swap = param[1].astype(bool)
        self.invert = param[-2:]

        # load gene filters
        self.filtering = np.load(os.path.join(self.storage_path, 'filtering.npy'))
        self.filter_genes = np.load(os.path.join(self.storage_path, 'filter_genes.npy'))
        self.selected_gene_idx = np.load(os.path.join(self.storage_path, 'selected_gene_idx.npy'))

    def plot_bleeding(self, gene, cmap='jet', save=False):
        '''
        Plot the raw reads, effective reads, and bleeding (if there is any) of a given gene
        where gene can be selected either by gene name or gene index
        '''
        if isinstance(gene, int):
            gene_idx = gene
        elif isinstance(gene, str):
            gene_idx = np.argwhere(self.gene_names == gene)[0][0]
        else:
            raise Exception('`gene` should be either a gene name(str) or the index of some gene(int)')
        logger.info('Gene: {}'.format(self.gene_names[gene_idx]))
        # load raw reads
        raw_count = np.load(self.storage_path + 'raw_count.npy')[:, self.selected_gene_idx[gene_idx]]
        pos = np.load(self.storage_path + 'all_spots_position.npy')
        raw_filtered_align = (raw_count[self.tissue_mask] == self.Reads[:, gene_idx]).sum()
        # determine if any bleeding filtering is performed
        if raw_filtered_align == self.n_spot_in:
            logger.info('\t no bleeding filtering performed')
        # calculate bleeding ratio
        all_counts = raw_count.sum()
        tissue_counts = self.Reads[:, gene_idx].sum()
        bleed_ratio = 1 - tissue_counts / all_counts
        logger.info('\t {:.3f}% bleeds out'.format(bleed_ratio * 100))

        # plot
        plot_intissue = np.ones_like(raw_count) * np.nan
        plot_intissue[self.tissue_mask] = self.Reads[:, gene_idx]
        plot_outside = raw_count.copy().astype(float)
        plot_outside[self.tissue_mask] = np.nan
        if bleed_ratio == 0:
            plot_data = np.vstack([raw_count, plot_intissue])
            plot_titles = ['Raw Read', 'Reads']
        else:
            plot_data = np.vstack([raw_count, plot_intissue, plot_outside])
            plot_titles = ['Raw Read', 'Reads', 'Bleeding']
        v_min = np.nanpercentile(plot_data, 5, axis=1)
        v_max = np.nanpercentile(plot_data, 95, axis=1)
        if self.layout == 1:
            marker = 'H'
            size = 5
        else:
            marker = 's'
            size = 10
        if save:
            save = self.storage_path + 'gene_bleeding_plots/'
            if not os.path.isdir(save):
                os.mkdir(save)
        logger.info(plot_data.shape)
        bp.st_plot(plot_data[:, None], pos, unit_dist=size, cmap=cmap, layout=marker, x_y_swap=self.x_y_swap,
                   invert=self.invert, v_min=v_min,
                   v_max=v_max, subtitles=plot_titles, name='{}_bleeding_plot'.format(self.gene_names[gene_idx]),
                   save=save)

    def bleeding_correction(self, n_top=50, max_steps=5, n_gene=None):
        cleaned_stdata = CleanedSTData(stdata=self, n_top=n_top, max_steps=max_steps)
        cleaned_stdata.load_data(self.data_name)

    def k_fold(self, cluster_storage, n_fold=5, n_splits=15, n_samples=100, n_burn=2000, n_thin=5, lda=0):
        return CrossValidationSTData(stdata=self, n_fold=n_fold, n_splits=n_splits,
                                     n_samples=n_samples, n_burn=n_burn, n_thin=n_thin, lda=lda)


class CleanedSTData(RawSTData):
    def __init__(self, stdata=None, n_top=50, max_steps=5, load_path=None):
        self.stdata = stdata
        self.n_top = n_top
        self.max_steps = max_steps
        self.load_path = load_path

    def load_data(self, data_name):
        self.data_name = data_name
        if self.load_path:
            self.load_cleaned(self.load_path)
        else:
            super().__init__(self.stdata.data_name, load=self.stdata.storage_path)
            self.positions = np.load(os.path.join(self.storage_path, 'all_spots_position.npy')).T.astype(int)
            self.raw_Reads = np.load(os.path.join(self.storage_path, 'raw_count.npy'))[:, self.selected_gene_idx]
        self.clean_data_plots = os.path.join(self.storage_path, 'cleaned_data_plots/')
        if not os.path.isdir(self.clean_data_plots):
            os.mkdir(self.clean_data_plots)

    def clean_bleed(self, n_top=50, max_steps=5, n_gene=None, local_weight=15):
        if not n_gene:
            n_gene = self.n_gene
        basis_idxs, basis_mask = bleed.build_basis_indices(self.positions, self.tissue_mask)

        if not self.has_non_tissue_spots():
            raise RuntimeError('Cannot run clean bleed without non-tissue spots.')

        self.global_rates, fit_Rates, self.basis_functions, self.Weights, _, _ = bleed.decontaminate_spots(
            self.raw_Reads[:, :n_gene], self.tissue_mask, basis_idxs, basis_mask, n_top=n_top, max_steps=max_steps,
            local_weight=local_weight)
        self.corrected_Reads = np.round(
            fit_Rates / fit_Rates.sum(axis=0, keepdims=True) * self.raw_Reads[:, :n_gene].sum(axis=0, keepdims=True))
        self.Reads = self.corrected_Reads[self.tissue_mask]

    def has_non_tissue_spots(self):
        return not np.all(self.tissue_mask)

    def get_suggested_initial_local_weight(self):
        return math.sqrt(self.tissue_mask.sum())

    def plot_basis_functions(self):
        basis_types = ['Out-Tissue', 'In-Tissue']
        basis_names = ['North', 'South', 'West', 'East']

        labels = [(d + t) for t in basis_types for d in basis_names]

        for d in range(self.basis_functions.shape[0]):
            plt.plot(np.arange(self.basis_functions.shape[1]), self.basis_functions[d], label=labels[d])
        plt.xlabel('Distance along cardinal direction')
        plt.ylabel('Relative bleed probability')
        plt.legend(loc='upper right')
        plt.savefig(self.clean_data_plots + 'A1_basis_functions.pdf', bbox_inches='tight')
        plt.close()

    def plot_before_after_cleanup(self, gene, cmap='jet', save=False):
        if isinstance(gene, int):
            gene_idx = gene
        elif isinstance(gene, str):
            gene_idx = np.argwhere(self.gene_names == gene)[0][0]
        else:
            raise Exception('`gene` should be either a gene name(str) or the index of some gene(int)')
        logger.info('Gene: {}'.format(self.gene_names[gene_idx]))

        # plot
        plot_data = np.vstack([self.raw_Reads[:, gene_idx], self.corrected_Reads[:, gene_idx]])
        plot_titles = ['Raw Read', 'Corrected Reads']
        v_min = np.nanpercentile(plot_data, 5, axis=1)
        v_max = np.nanpercentile(plot_data, 95, axis=1)
        if self.layout == 1:
            marker = 'H'
            size = 5
        else:
            marker = 's'
            size = 10
        if save:
            save = self.clean_data_plots + 'gene_bleeding_plots/'
            if not os.path.isdir(save):
                os.mkdir(save)
        bp.st_plot(plot_data, self.positions.T, unit_dist=size, cmap=cmap, layout=marker, x_y_swap=self.x_y_swap,
                   invert=self.invert, v_min=v_min, v_max=v_max, subtitles=plot_titles,
                   name='{}_bleeding_plot'.format(self.gene_names[gene_idx]), save=save)

    def save(self):
        np.save(os.path.join(self.storage_path, 'corrected_Reads'), self.corrected_Reads)
        np.save(os.path.join(self.storage_path, 'Reads'), self.Reads)
        np.save(os.path.join(self.storage_path, 'global_rates'), self.global_rates)
        np.save(os.path.join(self.storage_path, 'basis_functions'), self.basis_functions)
        np.save(os.path.join(self.storage_path, 'Weights'), self.Weights)

    def load_cleaned(self, load_path):
        super().load(load_path)
        self.corrected_Reads = np.load(os.path.join(self.storage_path, 'corrected_Reads.npy'))
        self.global_rates = np.load(os.path.join(load_path, 'global_rates.npy'))
        self.basis_functions = np.load(os.path.join(load_path, 'basis_functions.npy'))
        self.Weights = np.load(os.path.join(load_path, 'Weights.npy'))
        self.positions = np.load(os.path.join(self.storage_path, 'all_spots_position.npy')).T
        self.raw_Reads = np.load(os.path.join(self.storage_path, 'raw_count.npy'))[:, self.selected_gene_idx]


LSF_CV_JOB_TEMPLATE = """
#!/usr/bin/env bash
#BSUB -W {runtime_hours}:00
#BSUB -R rusage[mem={memory}]
#BSUB -J BayesTME[1-{n_jobs}]
#BSUB -e "{data_dir}/BayesTME_%I.err"
#BSUB -eo "{data_dir}/BayesTME_%I.out"

CONFIGPATH="k_fold/setup/config/BayesTME/config_${{LSB_JOBINDEX}}.cfg"

/opt/local/singularity/3.7.1/bin/singularity exec \
	--bind "{data_dir}":/data \
	/home/quinnj2/bayestme_latest.sif \
	grid_search \
	--data-dir "/data/k_fold/jobs/data" \
	--config "/data/${{CONFIGPATH}}" \
	--output-dir "/data/k_fold/results"
"""


class CrossValidationSTData(RawSTData):
    def __init__(self, stdata,
                 n_fold=5,
                 n_splits=15,
                 n_samples=100,
                 n_burn=2000,
                 n_thin=5,
                 lda=0,
                 n_comp_min=2,
                 n_comp_max=12,
                 lambda_values=(1, 1e1, 1e2, 1e3, 1e4, 1e5),
                 max_ncell=120):
        '''
        @param stdata:
        @param cluster_storage:
        @param n_fold:
        @param n_splits:
        @param n_samples:
        @param n_burn:
        @param n_thin:
        @param lda:
        @param n_comp_min:
        @param n_comp_max:
        @param lambda_values:
        @param max_ncell:
        '''
        super().__init__(stdata.data_name, stdata.storage_path)
        self.max_ncell = max_ncell
        self.lams = lambda_values
        self.n_comp_max = n_comp_max
        self.n_comp_min = n_comp_min
        self.lda = lda
        self.n_thin = n_thin
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.n_splits = n_splits

        self.k_fold_path = os.path.join(self.storage_path, 'k_fold/')
        pathlib.Path(self.k_fold_path).mkdir(parents=True, exist_ok=True)

        self.k_fold_jobs = os.path.join(self.k_fold_path, 'jobs/')
        pathlib.Path(self.k_fold_jobs).mkdir(parents=True, exist_ok=True)

        self.k_fold_data = self.k_fold_jobs + 'data/'
        pathlib.Path(self.k_fold_data).mkdir(parents=True, exist_ok=True)

        setup_path = os.path.join(self.k_fold_path, 'setup/')
        pathlib.Path(setup_path).mkdir(parents=True, exist_ok=True)

        config_root = os.path.join(setup_path, 'config/')
        pathlib.Path(config_root).mkdir(parents=True, exist_ok=True)
        self.config_path = os.path.join(config_root, '{}/'.format(self.data_name))

        results_root = os.path.join(setup_path, 'results/')
        pathlib.Path(results_root).mkdir(parents=True, exist_ok=True)

        self.results_path = os.path.join(results_root, '{}/'.format(self.data_name))
        self.likelihood_path = os.path.join(results_root, '{}/likelihoods/'.format(self.data_name))
        logger.info('results at {}'.format(self.results_path))

        # job log/error outputs storage path
        self.outputs_path = os.path.join(setup_path, 'outputs')
        pathlib.Path(self.outputs_path).mkdir(parents=True, exist_ok=True)
        logger.info('log/error at {}'.format(self.outputs_path))

        pathlib.Path(self.config_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.results_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.likelihood_path).mkdir(parents=True, exist_ok=True)

        self.n_fold = n_fold
        self.exc_file = 'grid_search_cfg.py'

    def prepare_jobs(self):
        self.save_folds(n_fold=self.n_fold, n_splits=self.n_splits)
        self.create_lsf_jobs(
            cluster_storage=self.storage_path,
            n_samples=self.n_samples,
            n_burn=self.n_burn,
            n_thin=self.n_thin,
            lda=self.lda,
            n_comp_min=self.n_comp_min,
            n_comp_max=self.n_comp_max,
            lams=self.lams,
            max_ncell=self.max_ncell)

    @staticmethod
    def create_folds(n_spot_in: int,
                     positions_tissue: np.ndarray,
                     layout: int,
                     reads: np.ndarray,
                     n_fold=5,
                     n_splits=15):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        edges = utils.get_edges(positions_tissue, layout=layout)
        n_neighbours = np.zeros(n_spot_in)
        if layout == 1:
            edge_threshold = 5
        else:
            edge_threshold = 3
        for i in range(n_spot_in):
            n_neighbours[i] = (edges[:, 0] == i).sum() + (edges[:, 1] == i).sum()
        splits = kf.split(np.arange(n_spot_in)[n_neighbours > edge_threshold])

        for k in range(n_fold):
            _, heldout = next(splits)
            mask = np.array(
                [i in np.arange(n_spot_in)[n_neighbours > edge_threshold][heldout] for i in range(n_spot_in)])
            train = reads.copy()
            test = reads.copy()
            train[mask] = 0
            test[~mask] = 0
            yield mask, train.astype(int), test.astype(int), n_neighbours

    def save_folds(self, n_fold=5, n_splits=15):
        fig, ax = plt.subplots(1, n_fold, figsize=(6 * (n_fold + 1), 6))
        if n_fold == 1:
            ax = [ax]

        for k, (mask, train, test, n_neighbours) in enumerate(CrossValidationSTData.create_folds(
                n_spot_in=self.n_spot_in,
                positions_tissue=self.positions_tissue,
                layout=self.layout,
                reads=self.Reads,
                n_fold=n_fold,
                n_splits=n_splits)):
            np.save(os.path.join(self.k_fold_data, '{}_mask_fold{}'.format(self.data_name, k)), mask)
            np.save(os.path.join(self.k_fold_data, '{}_fold{}'.format(self.data_name, k)), train.astype(int))
            np.save(os.path.join(self.k_fold_data, '{}_test{}'.format(self.data_name, k)), test.astype(int))
            bp.plot_spots(ax[k], n_neighbours, self.positions_tissue, s=5, cmap='viridis')
            ax[k].scatter(self.positions_tissue[0, mask], self.positions_tissue[1, mask], s=5, c='r')
        plt.savefig(self.k_fold_path + '{}_masks.pdf'.format(self.data_name))
        np.save(self.k_fold_data + '{}_pos'.format(self.data_name), self.positions_tissue)

    def write_cgf(self, n_samples, n_burn, n_thin, lda, spatial, folds=5, n_comp_min=2, n_comp_max=12,
                  lams=(1, 1e1, 1e2, 1e3, 1e4, 1e5), max_ncell=120, n_genes=(1000,)):
        config = configparser.ConfigParser()

        config['setup'] = {
            'n_samples': n_samples,
            'n_burn': n_burn,
            'n_thin': n_thin,
            'exp_name': self.data_name,
            'lda': lda,
            'spatial': spatial,
            'max_ncell': max_ncell
        }

        idx = 1
        for n_fold in range(folds):
            for n_comp in range(n_comp_min, n_comp_max + 1, 1):
                for lam2 in lams:
                    for n_gene in n_genes:
                        config['exp'] = {
                            'lam_psi': lam2,
                            'n_components': n_comp,
                            'n_fold': n_fold,
                            'n_gene': n_gene
                        }

                        with open(os.path.join(self.config_path, 'config_{}.cfg'.format(idx)), 'w') as configfile:
                            config.write(configfile)
                        idx += 1
        logger.info('{} jobs generated'.format(idx - 1))
        logger.info('\t {} cv folds'.format(folds))
        logger.info(
            '\t {} n_comp grid: {}'.format(n_comp_max - n_comp_min + 1, np.arange(n_comp_min, n_comp_max + 1, 1)))
        logger.info('\t {} lambda grid: {}'.format(len(lams), lams))
        logger.info('\t {} n_gene grid: {}'.format(len(n_genes), n_genes))
        return idx - 1

    def create_lsf_jobs(self,
                        cluster_storage,
                        n_samples=100,
                        n_burn=2000,
                        n_thin=5,
                        lda=0,
                        time_limit=96,
                        mem_req=24,
                        n_comp_min=2,
                        n_comp_max=12,
                        lams=(1, 1e1, 1e2, 1e3, 1e4, 1e5),
                        max_ncell=120):
        n_exp = self.write_cgf(
            n_samples, n_burn, n_thin,
            lda=lda,
            spatial=self.layout,
            folds=self.n_fold,
            n_genes=[self.n_gene],
            n_comp_min=n_comp_min,
            n_comp_max=n_comp_max,
            lams=lams,
            max_ncell=max_ncell)
        jobsfile = os.path.join(self.k_fold_jobs, '{}.sh'.format(self.data_name))

        with open(jobsfile, 'w') as f:
            f.write(
                LSF_CV_JOB_TEMPLATE.format(
                    data_dir=cluster_storage,
                    n_jobs=n_exp,
                    runtime_hours=time_limit,
                    memory=mem_req))


class DeconvolvedSTData(RawSTData):
    def __init__(self, load_path=None, stdata=None, cell_prob_trace=None, expression_trace=None, beta_trace=None,
                 cell_num_trace=None, lam=None):
        super().__init__(stdata.data_name, load=stdata.storage_path)
        self.results_path = self.storage_path + 'results/'
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
        if load_path:
            self.load_deconvolved(load_path)
        else:
            self.cell_prob_trace = cell_prob_trace
            self.expression_trace = expression_trace
            self.beta_trace = beta_trace
            self.cell_num_trace = cell_num_trace
            self.lam = lam
            self.n_components = self.expression_trace.shape[1]
            self.save_deconvolved()

    def detect_communities(self, min_clusters, max_clusters, assignments_ref=None, alignment=False):
        best_clusters, best_assignments, scores = communities.communities_from_posteriors(
            self.cell_prob_trace[:, :, 1:],
            self.edges, min_clusters=min_clusters,
            max_clusters=max_clusters,
            cluster_score=communities.gaussian_aicc_bic_mixture)
        if alignment and assignments_ref is not None:
            best_assignments, _, _ = communities.align_clusters(assignments_ref, best_assignments)
        return best_assignments

    def detect_marker_genes(self):
        score = (self.expression_trace == np.amax(self.expression_trace, axis=1)[:, None]).sum(axis=0)
        score /= self.expression_trace.shape[0]
        marker_gene = [self.features[score[k] > 0.95] for k in range(self.n_components)]
        return marker_gene

    def plot_deconvolution(self, plot_type='cell_prob', cmap='jet', seperate_pdf=False):
        '''
        plot the deconvolution results
        '''
        if self.layout == 1:
            marker = 'H'
            size = 5
        else:
            marker = 's'
            size = 10

        if plot_type == 'cell_prob':
            plot_object = self.cell_prob_trace[:, :, 1:].mean(axis=0)
        elif plot_type == 'cell_num':
            plot_object = self.cell_num_trace[:, :, 1:].mean(axis=0)
        else:
            raise Exception(
                "'plot_type' can only be either 'cell_num' for cell number or 'cell_prob' for cell-type probability")

        if seperate_pdf:
            for i in range(self.n_components):
                bp.st_plot(plot_object[:, i].T[:, None], self.positions_tissue, unit_dist=size, cmap=cmap,
                           x_y_swap=self.x_y_swap, invert=self.invert)
        else:
            bp.st_plot(plot_object.T[:, None], self.positions_tissue, unit_dist=size, cmap=cmap, x_y_swap=self.x_y_swap,
                       invert=self.invert)

    def plot_marker_genes(self, n_top=5):
        gene_expression = self.expression_trace.mean(axis=0)
        difference = np.zeros_like(gene_expression)
        n_components = gene_expression.shape[0]
        for k in range(n_components):
            max_exp = gene_expression.max(axis=0)
            difference[k] = gene_expression[k] / max_exp

        fig, ax = plt.subplots(1, 1, figsize=(8, 20))
        for i in range(self.n_components):
            ax.barh(np.arange(n_top * self.n_components)[::-1] + 0.35 - i * 0.1, gene_expression[i][marker_gene_idx],
                    height=0.1, label='cell_type{}'.format(i))
        for i in range(self.n_components - 1):
            ax.axhline(ref_gene.flatten().shape[0] / 7 * (i + 1) - 0.45, ls='--', alpha=0.5)
        #     ax.axvline(0)
        ax.set_yticks(np.arange(ref_gene.flatten().shape[0])[::-1])
        ax.set_yticklabels(ref_gene.flatten(), fontsize=20)
        #     ax.set_xlim(-0.01, 0.02)
        ax.margins(x=0.1, y=0.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc=4, fontsize=12)
        ax.set_title('Marker genes from the filtered results', fontsize=20)
        plt.tight_layout()
        plt.savefig('marker_gene_filtered.pdf')
        plt.close()

    def save_deconvolved(self):
        np.save(self.results_path + 'cell_prob_trace.npy', self.cell_prob_trace)
        np.save(self.results_path + 'expression_trace.npy', self.expression_trace)
        np.save(self.results_path + 'beta_trace.npy', self.beta_trace)
        np.save(self.results_path + 'cell_num_trace.npy', self.cell_num_trace)
        np.save(self.results_path + 'lam.npy', np.array([self.lam]))
        logger.info('Saved to {}'.format(self.results_path))

    def load_deconvolved(self, load_path):
        logger.info('Loading deconvolution results from {}'.format(load_path))
        self.cell_prob_trace = np.load(load_path + 'cell_prob_trace.npy')
        self.expression_trace = np.load(load_path + 'expression_trace.npy')
        self.beta_trace = np.load(load_path + 'beta_trace.npy')
        self.cell_num_trace = np.load(load_path + 'cell_num_trace.npy')
        self.lam = np.load(load_path + 'lam.npy')[0]
        self.n_components = self.expression_trace.shape[1]


class SpatialExpression(DeconvolvedSTData):
    def __init__(self, load_path=None, stdata=None,
                 n_spatial_patterns=10,
                 n_samples=100,
                 n_burn=100,
                 n_thin=5,
                 n_gene=50,
                 simple=False):
        super().__init__(load_path=stdata.results_path, stdata=stdata)
        self.spatial_path = self.results_path + 'spatial/'
        if not os.path.isdir(self.spatial_path):
            os.mkdir(self.spatial_path)
        if load_path:
            self.load_spatialexp(load_path)
        else:
            self.spatial_inference(
                n_spatial_patterns=n_spatial_patterns,
                n_samples=n_samples,
                n_burn=n_burn,
                n_thin=n_thin,
                n_gene=n_gene,
                simple=simple)
            self.save_spatialexp()

    def spatial_inference(self,
                          n_spatial_patterns=10,
                          n_samples=100,
                          n_burn=100,
                          n_thin=5,
                          n_gene=50,
                          simple=False):
        observed = utils.filter_reads_to_top_n_genes(reads=self.Reads, n_gene=n_gene)
        self.SDE = SpatialDifferentialExpression(
            n_cell_types=self.n_components,
            n_spatial_patterns=n_spatial_patterns,
            Obs=observed,
            edges=self.edges)

        read_trace = np.load(self.results_path + 'reads_trace.npy')
        self.SDE.spatial_detection(self.cell_num_trace, self.beta_trace, self.expression_trace, read_trace,
                                   n_samples=n_samples, n_burn=n_burn, n_thin=n_thin, ncell_min=2, simple=simple)

    def save_spatialexp(self):
        np.save(self.spatial_path + 'W_samples', self.SDE.W_samples)
        np.save(self.spatial_path + 'C_samples', self.SDE.C_samples)
        np.save(self.spatial_path + 'Gamma_samples', self.SDE.Gamma_samples)
        np.save(self.spatial_path + 'H_samples', self.SDE.H_samples)
        np.save(self.spatial_path + 'V_samples', self.SDE.V_samples)
        np.save(self.spatial_path + 'Theta_samples', self.SDE.Theta_samples)
        logger.info('Saved to {}'.format(self.spatial_path))

    def load_spatialexp(self, load_path):
        logger.info('Loading spatial results from {}'.format(load_path))
        self.W_samples = np.load(self.spatial_path + 'W_samples.npy')
        self.C_samples = np.load(self.spatial_path + 'C_samples.npy')
        self.Gamma_samples = np.load(self.spatial_path + 'Gamma_samples.npy')
        self.H_samples = np.load(self.spatial_path + 'H_samples.npy')
        self.V_samples = np.load(self.spatial_path + 'V_samples.npy')
        self.Theta_samples = np.load(self.spatial_path + 'Theta_samples.npy')

    def spatial_genes(self, h_threshold=0.95, magnitude_filter=None):
        score = (self.H_samples > 0).mean(axis=0)
        spatial_gene = []
        for gene, cell_type in np.argwhere(score > 0.95):
            exp_gene = self.Theta_samples[:, :, gene, cell_type].mean(axis=0)
            if magnitude_filter:
                if exp_gene.max() - exp_gene.min() > magnitude_filter:
                    spatial_gene.append([gene, cell_type])
            else:
                spatial_gene.append([gene, cell_type])
        self.spatial_gene = np.array(spatial_gene)

    def plot_spatial_patterns(self):
        spatial_plots_path = self.spatial_path + 'plots/'
        if not os.path.isdir(self.spatial_path):
            os.mkdir(self.spatial_path)
        cw_cmap = plt.get_cmap("coolwarm")
        modes, counts = mode(self.H_samples, axis=0)
        self.spatial_genes()
        for k in range(self.n_components):
            logger.info('cell type {}'.format(k))
            unique_genes = np.unique(self.spatial_gene[self.spatial_gene[:, 1] == k, 0])
            patterns = np.unique(modes[0, :, k], axis=0)
            patterns = patterns[patterns != 0]
            if len(patterns) > 0:
                f = 0
                for h in patterns:
                    gene_ids = np.argwhere(modes[0, unique_genes, k] == h)
                    if len(gene_ids) != 0 and abs(pearsonr(self.cell_num_trace[:, :, k + 1].mean(axis=0),
                                                           self.W_samples[:, k, h].mean(axis=0))[0]) < 0.5:
                        f += 1
                        loadings = self.V_samples[:, unique_genes[gene_ids], k].mean(axis=0).flatten()
                        rank = loadings.argsort()
                        gene_idx = unique_genes[gene_ids].flatten()[rank]
                        genes = self.gene_names[gene_idx]
                        n_genes = len(genes)
                        logger.info('\t {} spatial genes'.format(n_genes))
                        loadings = self.V_samples[:, gene_idx, k].mean(axis=0)
                        max_loading = np.max(np.abs(loadings)) * np.sign(loadings[np.argmax(np.abs(loadings))])
                        loadings /= np.max(np.abs(loadings)) * np.sign(loadings[np.argmax(np.abs(loadings))])

                        W_plot = self.W_samples[:, k, h].mean(axis=0) * max_loading
                        vmin = min(-1e-4, W_plot.min())
                        vmax = max(1e-4, W_plot.max())
                        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                        logger.info(np.std(W_plot))
                        bp.st_plot(W_plot, pos[::-1], cmap='coolwarm', layout='s', norm=norm, unit_dist=10,
                                   name='spatial_pattern_cell_type_{}_{}'.format(k, f), save=spatial_plots_path)
                        if n_genes > 15:
                            genes_selected = np.argsort(np.abs(loadings))[::-1][:15]
                            loadings = loadings[genes_selected]
                            n_genes = 15
                        fig, ax = plt.subplots(1, 1, figsize=(5, n_genes / 2 + 1))
                        loading_plot = loadings[np.argsort(loadings)]
                        vmin = min(-1e-4, loading_plot.min())
                        vmax = max(1e-4, loading_plot.max())
                        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                        ax.barh(np.arange(n_genes), loading_plot, color=cw_cmap(norm(loading_plot)))
                        for i, v in enumerate(loading_plot):
                            if v > 0:
                                ax.text(v, i, '{:.1f}'.format(v), fontweight='bold', ha='left', va='center')
                            else:
                                ax.text(v, i, '{:.1f}'.format(v), fontweight='bold', ha='right', va='center')
                        ax.set_yticks(np.arange(n_genes))
                        ax.set_yticklabels(genes[np.argsort(loadings)])
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.spines["left"].set_visible(False)
                        ax.set_title('Loading', fontsize=20, fontweight='bold')
                        ax.yaxis.tick_right()
                        ax.set_xlim(-1.1, 1.1)
                        plt.tight_layout()
                        plt.savefig(self.spatial_plots_path + 'spatial_loading_cell_type_{}_{}.pdf'.format(k, f))
