import bayestme.data
import bayestme.expression_truth
import bayestme.marker_genes
from bayestme import utils, data
from bayestme.mcmc import deconvolution
import bayestme.synthetic_data


def test_deconvolve():
    (
        locations,
        tissue_mask,
        true_rates,
        true_counts,
        bleed_counts,
    ) = bayestme.synthetic_data.generate_simulated_bleeding_reads_data(
        n_rows=15, n_cols=15, n_genes=5
    )

    edges = utils.get_edges(locations[tissue_mask, :], 2)

    n_samples = 3
    lam2 = 1000
    n_components = 3
    n_nodes = tissue_mask.sum()
    n_gene = 3

    result: data.DeconvolutionResult = deconvolution.deconvolve(
        true_counts[tissue_mask],
        edges,
        n_gene=3,
        n_components=3,
        lam2=1000,
        n_samples=3,
        n_burnin=1,
        n_thin=1,
        bkg=False,
        lda=False,
    )

    assert result.lam2 == lam2
    assert result.n_components == n_components
    assert result.cell_prob_trace.shape == (n_samples, n_nodes, n_components)
    assert result.cell_num_trace.shape == (n_samples, n_nodes, n_components)
    assert result.expression_trace.shape == (n_samples, n_components, n_gene)
    assert result.beta_trace.shape == (n_samples, n_components)
    assert result.reads_trace.shape == (n_samples, n_nodes, n_gene, n_components)
