import numpy as np
import anndata
import statsmodels.api as sm


def get_rank(array):
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def generate_semi_synthetic(adata: anndata.AnnData,
                            cluster_id_column: str,
                            pos_ss: np.ndarray,
                            n_genes: int,
                            cell_num=None,
                            canvas_size=(36, 36),
                            sq_size=4,
                            layout=None,
                            random_seed=None,
                            n_spatial_gene=50,
                            alpha=1,
                            w=None,
                            spatial_gene=None,
                            spatial_cell_type=None,
                            spatial_programs=None,
                            verbose=True):
    if verbose:
        print('Generating semi-synthetic data...')
        if cell_num is not None:
            print('\t with cell num. annotation at {} spots'.format(cell_num.shape[0]))
    if spatial_gene is None or spatial_cell_type is None:
        print('Picking sptial genes and cell types by dispersion')
    else:
        print('Using given sptial genes and cell types')
    if random_seed is not None:
        np.random.seed(random_seed)
    # get single nucleus reads
    adata_snrna_raw = adata

    # get the cell type list correspond to the count matrix
    cell_types = adata_snrna_raw.obs[cluster_id_column]

    # filter the top n_genes most expressed genes
    try:
        gene_count_mat = np.asarray(adata_snrna_raw.X.todense())
    except AttributeError:
        gene_count_mat = adata_snrna_raw.X

    top = np.argsort(np.std(np.log(1 + gene_count_mat), axis=0))[::-1]
    gene_count_mat = gene_count_mat[:, top[:n_genes]].astype(int)

    # filter the selected cells
    classes = []
    for cell_type in cell_types.unique():
        classes.append(np.where(cell_types == cell_type)[0])

    # generate the synthetic celluar community indices
    if layout is not None:
        bkg = layout
        n_sectors = np.unique(bkg.flatten()).size
    else:
        bkg = np.zeros((canvas_size[0], canvas_size[1]), dtype=int)
        n_row = int(canvas_size[0] / sq_size)
        n_col = int(canvas_size[1] / sq_size)
        n_sectors = 0
        for i in range(n_row):
            for j in range(n_col):
                bkg[i * sq_size:(i + 1) * sq_size, j * sq_size:(j + 1) * sq_size] = n_sectors
                n_sectors = n_sectors + 1

    # assign celluar community indices to spots in tissue
    prior_idx = np.zeros(pos_ss.shape[1])
    # assign each spot on the tissue to a region
    for i in range(pos_ss.shape[1]):
        x = pos_ss[0, i]
        y = pos_ss[1, i]
        prior_idx[i] = bkg[x, y]
    prior_idx = prior_idx.astype(int)

    # generate synthetic cell type probilities
    n_components = len(classes)
    priors = np.random.dirichlet(np.ones(n_components) * alpha, size=bkg.max() + 1)
    priors /= priors.sum(axis=1, keepdims=True)
    Truth_prior = priors[prior_idx]

    n_nodes = pos_ss.shape[1]
    # get the true rate of total number of cells in each region
    # total num of cells in spot i ~ Pois(lambda_i)
    lambdas = np.random.gamma(90, 1 / 3, n_sectors)
    n_cells = np.zeros((n_nodes, n_components), dtype=int)
    for i in range(n_nodes):
        if i % 1000 == 0:
            print(i)
        n_cells[i] = np.random.multinomial(np.random.poisson(lambdas[prior_idx[i]]), Truth_prior[i])

    # sample cells from scRNA data
    # sampled cells for each cell type
    sampled_cells = []
    # spot location of each sampled cells for each cell type
    sampled_cells_spots = []
    # UMI counts of each sampled cells for each cell type
    sampled_cell_reads = []
    for k in range(n_components):
        # randomly choose cells for each cell type
        sampled_cells_k = np.random.choice(classes[k], n_cells[:, k].sum()).astype(int)
        sampled_cells_spots_k = np.zeros(n_cells[:, k].sum()).astype(int)
        current_spot = 0
        for i in range(n_nodes):
            # assign each sampled cell to some spots
            sampled_cells_spots_k[current_spot:current_spot + n_cells[i, k]] = i
            current_spot += n_cells[i, k]
        sampled_cells.append(sampled_cells_k)
        sampled_cells_spots.append(sampled_cells_spots_k)
        sampled_cell_reads.append(gene_count_mat[sampled_cells_k])

    if n_spatial_gene > 0:
        if w is None:
            w = np.zeros((n_nodes, 3))
            w[:, 0] = np.cos(1 / 4 * pos_ss[0]) * 10
            w[:, 1] = np.sin(1 / 10 * pos_ss[1]) * 10
            w[:, 2] = np.cos(1 / 600 * pos_ss[0] * pos_ss[1]) * 10
        n_spatial_programs = w.shape[1]

        if spatial_gene is None or spatial_cell_type is None:
            # calculate dispersion of genes for each cell type
            alphas = np.zeros((n_genes, n_components))
            for k in range(n_components):
                for i in range(n_genes):
                    if (sampled_cell_reads[k][:, i] > 0).sum() > int(sampled_cell_reads[k].shape[0] / 2):
                        x = np.ones(sampled_cell_reads[k].shape[0])
                        res = sm.NegativeBinomial(sampled_cell_reads[k][:, i], x, loglike_method='nb2').fit()
                        alphas[i, k] = res.params[1]

            spatial_gene, spatial_cell_type = np.unravel_index(np.argsort(alphas, axis=None)[::-1], alphas.shape)

        # assign each spatial gene with some spatial program
        if spatial_programs is None:
            spatial_programs = np.zeros(n_spatial_gene).astype(int)
            for i in range(n_spatial_gene):
                spatial_programs[i] = np.random.randint(n_spatial_programs)

        for i in range(n_spatial_gene):
            expression_current_gene = np.random.normal(
                w[:, spatial_programs[i]][sampled_cells_spots[spatial_cell_type[i]]])
            expression_rank = get_rank(expression_current_gene)

            # reorder the sampled gene
            # sampled_cell_reads[spatial_cell_type[i]][:, spatial_gene[i]] = np.sort(sampled_cell_reads[spatial_cell_type[i]][:, spatial_gene[i]])[expression_rank]

            # genereate poisson synthetic reads
            gene_lam = sampled_cell_reads[spatial_cell_type[i]][:, spatial_gene[i]].mean()
            syn_gene_reads = np.random.poisson(gene_lam, size=len(sampled_cell_reads[spatial_cell_type[i]]))
            sampled_cell_reads[spatial_cell_type[i]][:, spatial_gene[i]] = np.sort(syn_gene_reads)[expression_rank]
        spatial = np.array([spatial_gene[:n_spatial_gene], spatial_cell_type[:n_spatial_gene], spatial_programs])
    else:
        spatial = None

    Observation = np.zeros((n_nodes, n_genes, n_components))
    for i in range(n_nodes):
        for k in range(n_components):
            Observation[i, :, k] = sampled_cell_reads[k][sampled_cells_spots[k] == i].sum(axis=0)
    Observations_tissue = Observation.sum(axis=-1).astype(int)

    return Observations_tissue, Observation, Truth_prior, n_cells, spatial, sampled_cell_reads
