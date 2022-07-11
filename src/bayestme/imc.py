import os
import pandas as pd
import argparse
import logging

import numpy as np
from bayestme.gfbt_multinomial import GraphFusedBinomialTree

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='IMC with label')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--sc-file', type=str,
                    help='sc file')
parser.add_argument('--n-samples', type=int,
                    help='n samples',
                    default=200)
parser.add_argument('--n-burn', type=int,
                    help='n burn',
                    default=500)
parser.add_argument('--n-thin', type=int,
                    help='n thin',
                    default=5)


class IMCWithLabel:
    def __init__(self, n_components, edges, Observations, n_gene=300, n_max=120, background_noise=False, random_seed=0,
                 mask=None,
                 c=4, D=30, tf_order_psi=0, lam_psi=1e-2, lda_initialization=False, known_cell_num=None,
                 known_spots=None,
                 Truth_expression=None, Truth_prob=None, Truth_cellnum=None, Truth_reads=None, Truth_beta=None,
                 **kwargs):
        # just need n_components, edges, and Observations
        # n_components = labels.max() + 1
        # Observation = labels
        np.random.seed(random_seed)
        self.n_components = n_components
        self.n_max = n_max
        self.n_gene = n_gene
        self.edges = edges
        self.bkg = background_noise
        self.gtf_psi = GraphFusedBinomialTree(self.n_components + 1, edges, lam2=lam_psi)
        self.mask = mask
        self.n_nodes = self.gtf_psi.n_nodes
        self.cell_num = np.zeros((Observations.shape[0], Observations.max() + 2))
        self.cell_num[np.arange(Observations.shape[0]), Observations + 1] = 1
        self.cell_num[:, 0] = 1
        self.probs = self.cell_num

    def sample_probs(self, mult):
        '''
        sample cell-type probability psi_ik with spatial smoothing
        '''
        # clean up the GFTB input cell num
        # if self.bkg:
        #     cell_num = self.cell_num[:, :-1].copy()
        # else:
        cell_num = self.cell_num.copy()
        cell_num[:, 0] = 1 - cell_num[:, 0]
        cell_num *= mult
        self.gtf_psi.resample(cell_num)
        self.probs = self.gtf_psi.probs
        self.probs[:, 0] = 1 - self.probs[:, 0]
        self.probs[:, 1:] /= self.probs[:, 1:].sum(axis=1, keepdims=True)

    def sample(self, mult):
        self.sample_probs(mult)


def get_cell_prob_trace(img_id, edges, labels, pos, save_path,
                        n_samples=200,
                        n_thin=5,
                        n_burn=500):
    n_nodes = labels.shape[0]
    n_components = labels.max() + 1

    gfnb = IMCWithLabel(n_components=n_components, edges=edges, Observations=labels, lam=1000)

    cell_prob_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    for step in range(n_samples * n_thin + n_burn):
        if step % 10 == 0:
            logger.info(f'Step {step}')
        gfnb.sample(100)
        if step >= n_burn and (step - n_burn) % n_thin == 0:
            idx = (step - n_burn) // n_thin
            cell_prob_trace[idx] = gfnb.probs

    np.save(os.path.join(save_path, '{}_prob_trace'.format(img_id)), cell_prob_trace)


def get_job_index():
    return int(os.environ['LSB_JOBINDEX'])


def main():
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    
    # setting paths
    node_path = os.path.join(args.data_dir, 'nodes_celltype')
    edge_path = os.path.join(args.data_dir, 'edges_knn_10')
    coord_path = os.path.join(args.data_dir, 'coords')
    prob_trace_path = os.path.join(args.data_dir, 'cell_prob_trace')

    sc_df = pd.read_csv(args.sc_file)
    celltypes = list(set(sc_df["cellPhenotype"]))

    # calculate by image index
    sc_df_grouped = sc_df.groupby(['ImageNumber'])

    group_keys = sorted(sc_df_grouped.groups.keys())

    idx = get_job_index()
    current_image_id = group_keys[idx]
    sub_data = sc_df_grouped.get_group(current_image_id)

    logger.info('running image {}...'.format(current_image_id))
    labels = np.array([celltypes.index(c) for c in sub_data['cellPhenotype']], dtype=np.int8)
    edges = np.load(os.path.join(edge_path, '{}.npy'.format(current_image_id)))
    pos = np.load(os.path.join(coord_path, '{}.npy'.format(current_image_id)))
    get_cell_prob_trace(current_image_id, edges, labels, pos, prob_trace_path,
                        n_samples=args.n_samples,
                        n_thin=args.n_thin,
                        n_burn=args.n_burn)
