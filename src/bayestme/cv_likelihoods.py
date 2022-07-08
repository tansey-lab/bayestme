import logging
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from bayestme import data

logger = logging.getLogger(__name__)


def load_likelihoods(output_dir):
    results = []
    fold_nums = []
    lam_vals = []
    k_vals = []
    for fn in glob.glob(os.path.join(output_dir, "fold_*.h5ad")):
        result = data.PhenotypeSelectionResult.read_h5(fn)
        fold_nums.append(result.fold_number)
        lam_vals.append(result.lam)
        k_vals.append(result.n_components)

        results.append(data.PhenotypeSelectionResult.read_h5(fn))

    fold_nums = sorted(set(fold_nums))
    lam_vals = sorted(set(lam_vals))
    k_vals = sorted(set(k_vals))

    logger.info(f'Folds: {fold_nums}')
    logger.info(f'Lambdas: {lam_vals}')
    logger.info(f'Ks: {k_vals}')

    likelihoods = np.full((2, len(k_vals), len(lam_vals), len(fold_nums)), np.nan)
    for result in results:
        kidx = k_vals.index(result.n_components)
        lamidx = lam_vals.index(result.lam)
        foldidx = fold_nums.index(result.fold_number)

        likelihoods[0, kidx, lamidx, foldidx] = np.nanmean(result.log_lh_train_trace)
        likelihoods[1, kidx, lamidx, foldidx] = np.nanmean(result.log_lh_test_trace)

    logger.info(f'{(~np.isnan(likelihoods)).sum()} non-missing')

    return likelihoods, fold_nums, lam_vals, k_vals


def plot_likelihoods(
        likelihood_path,
        out_file,
        exp_name='Experiment',
        normalize=False):
    (train_likelihoods, test_likelihoods), fold_nums, lam_vals, k_vals = load_likelihoods(likelihood_path)
    # Plot the averages for train and test
    like_means = []
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=1)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        colors = ['blue', 'orange', 'green', 'purple', 'red', 'teal']

        like_mean = np.nanmean(test_likelihoods, axis=(-2, -1))
        if normalize:
            like_mean = (like_mean - like_mean.mean()) / like_mean.std()
        like_means.append(like_mean)
        plt.plot(k_vals, like_mean, label=f'Score for {exp_name}', lw=3, color=colors[0])

        like_mean = np.nanmean(test_likelihoods, axis=(-2, -1))
        if normalize:
            like_mean = (like_mean - like_mean.mean()) / like_mean.std()
        plt.scatter(k_vals[np.nanargmax(like_mean)], np.nanmax(like_mean), color=colors[0], s=150, marker=(5, 1),
                    zorder=10, label=f'Peak for {exp_name}')
        plt.gca().set_xlabel('Number of cell types', fontsize=16, weight='bold')
        plt.gca().set_ylabel('Relative test likelihood' if normalize else 'Test Likelihood', fontsize=16, weight='bold')
        plt.gca().set_xticks(k_vals)
        plt.gca().set_xticklabels([str(k) for k in k_vals])
        plt.gca().set_xlim([k_vals[0], k_vals[-1]])
        for label in plt.gca().xaxis.get_ticklabels():
            label.set_horizontalalignment('center')
        plt.legend(loc='lower right', ncol=2)
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
    return np.array(like_means)


def plot_cv_running(results_path, out_path):
    likelihoods, fold_nums, lam_vals, k_vals = load_likelihoods(results_path)
    # Plot the averages for train and test
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=1)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        fig, axarr = plt.subplots(1, 2, figsize=(16, 5), sharex='all')
        for tidx, traintest in enumerate(['Train', 'Test']):
            for lamidx, lam in enumerate(lam_vals):
                axarr[tidx].plot(k_vals, np.nanmean(likelihoods[tidx, :, lamidx], axis=-1),
                                 label=f'$\\lambda={lam:.1f}$', lw=2, ls='dotted')
            like_min = np.nanmin(np.nanmean(likelihoods[tidx], axis=-1), axis=-1)
            like_max = np.nanmax(np.nanmean(likelihoods[tidx], axis=-1), axis=-1)
            like_mean = np.nanmean(likelihoods[tidx], axis=(-2, -1))
            if np.all(np.isnan(like_min)):
                continue
            axarr[tidx].plot(k_vals, like_min, label=f'$\\lambda$ min', lw=3, color='darkgray', alpha=0.5)
            axarr[tidx].plot(k_vals, like_max, label=f'$\\lambda$ max', lw=3, color='gray', alpha=0.5)
            axarr[tidx].plot(k_vals, like_mean, label=f'$\\lambda$ mean', lw=3, color='black')
            axarr[tidx].scatter(k_vals[np.nanargmax(like_min)], np.nanmax(like_min), color='magenta', s=150,
                                marker=(5, 1), zorder=10)
            axarr[tidx].scatter(k_vals[np.nanargmax(like_max)], np.nanmax(like_max), color='magenta', s=150,
                                marker=(5, 1), zorder=10)
            axarr[tidx].scatter(k_vals[np.nanargmax(like_mean)], np.nanmax(like_mean), color='magenta', s=150,
                                marker=(5, 1), zorder=10)
            title = traintest
            axarr[tidx].set_title(title, weight='bold', fontsize=18)
            axarr[tidx].set_xlabel('Number of cell types', fontsize=16, weight='bold')
            axarr[tidx].set_ylabel('Likelihood', fontsize=16, weight='bold')
            axarr[tidx].set_xticks(k_vals)
            axarr[tidx].set_xticklabels([str(k) for k in k_vals])
            axarr[tidx].set_xlim([k_vals[0], k_vals[-1]])
            for label in axarr[tidx].xaxis.get_ticklabels():
                label.set_horizontalalignment('center')

        axarr[0].legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f'cv_running.pdf'), bbox_inches='tight')
        plt.close()
        plt.figure()
        for lamidx, lam in enumerate(lam_vals):
            plt.gca().bar(np.array(k_vals) - 0.25 + 0.5 * lamidx / len(lam_vals),
                          (~np.isnan(likelihoods[1, :, lamidx])).sum(axis=-1), width=0.5 / len(lam_vals),
                          label=f'$\\lambda={lam:.1f}$')
        title = 'Folds finished'
        plt.gca().set_title(title, weight='bold', fontsize=18)
        plt.gca().set_xlabel('Number of cell types', fontsize=16, weight='bold')
        plt.gca().set_ylabel('Folds finished', fontsize=16, weight='bold')
        plt.gca().set_xticks(k_vals)
        plt.gca().set_xticklabels([str(k) for k in k_vals])
        plt.gca().set_xlim([k_vals[0] - 1, k_vals[-1] + 1])
        plt.gca().set_ylim([0, len(fold_nums)])

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f'k-folds.pdf'), bbox_inches='tight')
        plt.close()


def get_max_likelihood_n_components(likelihoods):
    likelihood_mean = np.nanmean(likelihoods[1], axis=(-2, -1))

    return np.argmax(likelihood_mean) + 2
