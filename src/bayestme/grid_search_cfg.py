import pprint
import numpy as np
import argparse
import configparser
import os
import pathlib
import logging

from scipy.stats import multinomial

from . import utils
from .model_bkg import GraphFusedMultinomial

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='GFMM modeling on st data')
parser.add_argument('--config', type=str, default='semi_syn_1.cfg',
                    help='configration file')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--output-dir', type=str,
                    help='output data dir')


def sample_graph_fused_multinomial(
        train: np.ndarray,
        test: np.ndarray,
        n_components,
        edges: np.ndarray,
        n_gene: int,
        lam_psi: float,
        background_noise: bool,
        lda_initialization: bool,
        mask: np.ndarray,
        n_max: int,
        n_samples: int,
        n_thin: int,
        n_burn: int):
    n_nodes = train.shape[0]

    heldout_spots = np.argwhere(mask).flatten()
    train_spots = np.argwhere(~mask).flatten()
    if len(heldout_spots) == 0:
        mask = None

    graph_fused_multinomial = GraphFusedMultinomial(
        n_components=n_components,
        edges=edges,
        observations=train,
        n_gene=n_gene,
        lam_psi=lam_psi,
        background_noise=background_noise,
        lda_initialization=lda_initialization,
        mask=mask,
        n_max=n_max)

    cell_prob_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    cell_num_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    expression_trace = np.zeros((n_samples, n_components, n_gene))
    beta_trace = np.zeros((n_samples, n_components))
    loglhtest_trace = np.zeros(n_samples)
    loglhtrain_trace = np.zeros(n_samples)

    for step in range(n_samples * n_thin + n_burn):
        if step % 10 == 0:
            logger.info(f'Step {step}')
        # perform Gibbs sampling
        graph_fused_multinomial.sample(train)
        # save the trace of GFMM parameters
        if step >= n_burn and (step - n_burn) % n_thin == 0:
            idx = (step - n_burn) // n_thin
            cell_prob_trace[idx] = graph_fused_multinomial.probs
            expression_trace[idx] = graph_fused_multinomial.phi
            beta_trace[idx] = graph_fused_multinomial.beta
            cell_num_trace[idx] = graph_fused_multinomial.cell_num
            rates = (graph_fused_multinomial.probs[:, 1:][:, :, None] * (
                    graph_fused_multinomial.beta[:, None] * graph_fused_multinomial.phi)[None])
            nb_probs = rates.sum(axis=1) / rates.sum(axis=(1, 2))[:, None]
            loglhtest_trace[idx] = np.array(
                [multinomial.logpmf(test[i], test[i].sum(), nb_probs[i]) for i in heldout_spots]).sum()
            loglhtrain_trace[idx] = np.array(
                [multinomial.logpmf(train[i], train[i].sum(), nb_probs[i]) for i in train_spots]).sum()
            logger.info('{}, {}'.format(loglhtrain_trace[idx], loglhtest_trace[idx]))

    return (
        cell_prob_trace,
        cell_num_trace,
        expression_trace,
        beta_trace,
        loglhtest_trace,
        loglhtrain_trace
    )


def run_sampling_from_config(config,
                             data_dir,
                             output_dir):
    lam_psi = float(config['exp']['lam_psi'])
    n_samples = int(config['setup']['n_samples'])
    n_burn = int(config['setup']['n_burn'])
    n_thin = int(config['setup']['n_thin'])
    n_components = int(config['exp']['n_components'])
    exp_name = config['setup']['exp_name']
    n_fold = int(config['exp']['n_fold'])
    spatial = int(config['setup']['spatial'])
    max_ncell = int(config['setup']['max_ncell'])
    n_gene_raw = int(config['exp']['n_gene'])

    pos_ss = np.load(os.path.join(data_dir, '{}_pos.npy'.format(exp_name)))
    test = np.load(os.path.join(data_dir, '{}_test{}.npy'.format(exp_name, n_fold)))
    train = np.load(os.path.join(data_dir, '{}_fold{}.npy'.format(exp_name, n_fold)))
    mask = np.load(os.path.join(data_dir, '{}_mask_fold{}.npy'.format(exp_name, n_fold)))
    n_gene = min(train.shape[1], n_gene_raw)
    top = np.argsort(np.std(np.log(1 + train), axis=0))[::-1]
    train = train[:, top[:n_gene]]
    test = test[:, top[:n_gene]]

    if spatial == 0:
        edges = utils.get_edges(pos_ss, layout=2)
    else:
        edges = utils.get_edges(pos_ss, layout=1)

    (
        cell_prob_trace,
        cell_num_trace,
        expression_trace,
        beta_trace,
        loglhtest_trace,
        loglhtrain_trace
    ) = sample_graph_fused_multinomial(
        train=train,
        test=test,
        n_components=n_components,
        edges=edges,
        n_gene=n_gene,
        lam_psi=lam_psi,
        background_noise=False,
        lda_initialization=False,
        mask=mask,
        n_max=max_ncell,
        n_samples=n_samples,
        n_thin=n_thin,
        n_burn=n_burn)

    pathlib.Path(os.path.join(output_dir, 'likelihoods')).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_dir,
                         '{}_{}_{}_cell_prob_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components,
                                                                  lam_psi, n_fold)), cell_prob_trace)
    np.save(os.path.join(output_dir,
                         '{}_{}_{}_phi_post_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components, lam_psi,
                                                                 n_fold)), expression_trace)
    np.save(os.path.join(output_dir,
                         '{}_{}_{}_beta_post_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components,
                                                                  lam_psi, n_fold)), beta_trace)
    np.save(os.path.join(output_dir,
                         '{}_{}_{}_cell_num_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components, lam_psi,
                                                                 n_fold)), cell_num_trace)
    np.save(os.path.join(output_dir,
                         'likelihoods/{}_{}_{}_train_likelihood_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell,
                                                                                     n_components, lam_psi, n_fold)),
            loglhtrain_trace)
    np.save(os.path.join(output_dir,
                         'likelihoods/{}_{}_{}_test_likelihood_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell,
                                                                                    n_components, lam_psi, n_fold)),
            loglhtest_trace)


def main():
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    config = configparser.ConfigParser()

    logger.info('Reading configuration from {}'.format(args.config))

    config.read(args.config)

    logger.info('Parsed configuration object: {}'.format(pprint.pformat(config)))

    run_sampling_from_config(config=config,
                             data_dir=args.data_dir,
                             output_dir=args.output_dir)


if __name__ == '__main__':
    main()
