import numpy as np


def adjacency_matrix_from_edges(edges):
    from scipy.sparse import coo_matrix
    '''Takes an N x 2 numpy array of edges and converts to an adjacency matrix format.'''
    # Get the maximum node ID
    n_nodes = np.max(edges) + 1

    data = np.ones(n_nodes + edges.shape[0] * 2)
    rows = np.concatenate([np.arange(n_nodes), edges[:, 0], edges[:, 1]])
    cols = np.concatenate([np.arange(n_nodes), edges[:, 1], edges[:, 0]])
    return coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def gaussian_bic(posteriors, assignments, clusters):
    n_samples = assignments.shape[0]
    dof = clusters.shape[0]
    sum_sqerr = ((posteriors.mean(axis=0) - clusters[assignments]) ** 2).sum()
    return dof * np.log(n_samples) + 2 * sum_sqerr


def gaussian_aicc(posteriors, assignments, clusters):
    n_samples = assignments.shape[0]
    dof = clusters.shape[0]
    sum_sqerr = ((posteriors.mean(axis=0) - clusters[assignments]) ** 2).sum()
    return dof * 2 + 2 * sum_sqerr  # + (2*dof**2 + 2*dof) / (n_samples - dof - 1)


def gaussian_aicc_bic_mixture(posteriors, assignments, clusters, bic_proportion=0.5):
    return gaussian_bic(posteriors, assignments, clusters) * bic_proportion + (1 - bic_proportion) * gaussian_aicc(
        posteriors, assignments, clusters)


def swap_labels(labels, perm):
    # Switch the labels appropriately
    temp = np.array(labels)
    for i, k in enumerate(perm):
        temp[labels == i] = k
    return temp


def exhaustive_best_permutation(source, target, n_clusters):
    from itertools import permutations
    best_score, best_assignments, best_perm = None, None, None
    for perm in permutations(range(n_clusters)):
        # Swap the assignments
        perm_assignments = swap_labels(source, perm)

        # Check the overlap
        score = (target == perm_assignments).sum()
        if best_score is None or score > best_score:
            best_score = score
            best_assignments = perm_assignments
            best_perm = perm
    return best_perm


def greedy_best_permutation(source, target, n_clusters):
    # Greedy selection, starting with the largest cluster
    _, frequency = np.unique(target, return_counts=True)
    best_perm = [None] * n_clusters
    for i in np.argsort(frequency)[::-1]:
        # Find the most common label in source when looking at the target cluster
        labels, counts = np.unique(source[target == i], return_counts=True)
        for j in labels[np.argsort(counts)[::-1]]:
            if j not in best_perm:
                best_perm[j] = i
                break

    # If we didn't successfully find any merges, just set things to arbitrary leftover cluster IDs
    for i in range(n_clusters):
        if best_perm[i] is None:
            for j in range(n_clusters):
                if j not in best_perm:
                    best_perm[i] = j
    return best_perm


def communities_from_posteriors_separate(posteriors, edges, n_clusters=None, min_clusters=1, max_clusters=7,
                                         cluster_score=gaussian_aicc_bic_mixture):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.stats import mode

    if n_clusters is None:
        # Choose the best cluster from a grid of options
        print(f'Choosing n clusters from {min_clusters} to {max_clusters}')
        scores = np.zeros(max_clusters - min_clusters + 1)
        cluster_options = list(range(min_clusters, max_clusters + 1))
        best_score = None
        for k in cluster_options:
            print(k)
            clusters, assignments = communities_from_posteriors(posteriors, edges, n_clusters=k)
            scores[k - min_clusters] = cluster_score(posteriors, assignments, clusters)
            if best_score is None or scores[k - min_clusters] < best_score:
                best_score = scores[k - min_clusters]
                best_clusters = clusters
                best_assignments = assignments
        print(scores)
        return best_clusters, best_assignments, scores

    # Get the constraint matrix
    adj = adjacency_matrix_from_edges(edges)

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=adj)

    # Cluster each posterior sample
    assignments = np.zeros(posteriors.shape[:2], dtype=int)
    clusters = np.zeros((posteriors.shape[0], n_clusters, posteriors.shape[-1]))

    # Start with the first sample
    assignments[0] = ward.fit_predict(posteriors[0])

    # Now cluster subsequent samples while aligning them to prevent label switching
    for t in range(1, posteriors.shape[0]):
        assignments[t] = ward.fit_predict(posteriors[t])

        if n_clusters < 9:
            # Try all permutations to see which one fits the best
            best_perm = exhaustive_best_permutation(assignments[t], assignments[t - 1], n_clusters)
        else:
            # Greedy selection, starting with the largest cluster
            best_perm = greedy_best_permutation(assignments[t], assignments[t - 1], n_clusters)
        best_assignments = swap_labels(assignments[t], best_perm)

        # Switch the labels appropriately
        assignments[t] = best_assignments

    # Get the most common cluster assignment
    # map_assignments = mode(assignments, axis=0)[0][0]

    # Cluster the clusters!
    map_assignments = ward.fit_predict(assignments.T)

    # Average the clusters to get the best centroids
    clusters = np.array([posteriors[:, map_assignments == i].mean(axis=0).mean(axis=0) for i in range(n_clusters)])

    return clusters, map_assignments


def communities_from_posteriors(posteriors, edges, n_clusters=None, min_clusters=1, max_clusters=20,
                                cluster_score=gaussian_aicc_bic_mixture):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.stats import mode

    if n_clusters is None:
        # Choose the best cluster from a grid of options
        print(f'Choosing n clusters from {min_clusters} to {max_clusters}')
        scores = np.zeros(max_clusters - min_clusters + 1)
        cluster_options = list(range(min_clusters, max_clusters + 1))
        best_score = None
        for k in cluster_options:
            print(k)
            clusters, assignments = communities_from_posteriors(posteriors, edges, n_clusters=k)
            scores[k - min_clusters] = cluster_score(posteriors, assignments, clusters)
            if best_score is None or scores[k - min_clusters] < best_score:
                best_score = scores[k - min_clusters]
                best_clusters = clusters
                best_assignments = assignments
        print(scores)
        return best_clusters, best_assignments, scores

    # Get the constraint matrix
    adj = adjacency_matrix_from_edges(edges)

    # Flatten the posteriors into a single vector for each point
    posteriors = np.transpose(posteriors, [1, 0, 2])

    # Cluster the points
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=adj)
    assignments = ward.fit_predict(posteriors.reshape(posteriors.shape[0], -1))

    # Average the clusters to get the best centroids
    clusters = np.array([posteriors[assignments == i].mean(axis=(0, 1)) for i in range(n_clusters)])

    return clusters, assignments


def align_clusters(assignments_ref, assignments):
    cluster_ref = []
    for i in range(len(np.unique(assignments_ref))):
        cluster_ref.append(np.argwhere(assignments_ref == i).flatten())
    cluster = []
    for i in np.unique(assignments):
        cluster.append(np.argwhere(assignments == i).flatten())
    assignment = np.zeros(assignments.shape[0])
    if len(cluster_ref) <= len(cluster):
        for i, c_ref in enumerate(cluster_ref):
            best_overlap = 0
            for j, c in enumerate(cluster):
                intersect = np.intersect1d(c, c_ref)
                if len(intersect) > best_overlap:
                    best_c = c
                    best_overlap = len(intersect)
                    best_idx = j
            assignment[best_c] = i
            cluster.pop(best_idx)
        cluster_id = len(cluster_ref)
        for c in cluster:
            assignment[c] = cluster_id
            cluster_id += 1
    else:
        for i, c in enumerate(cluster):
            best_overlap = 0
            for j, c_ref in enumerate(cluster_ref):
                intersect = np.intersect1d(c, c_ref)
                if len(intersect) > best_overlap:
                    best_overlap = len(intersect)
                    best_idx = j
            assignment[c] = best_idx
    return assignment.astype(int), cluster_ref, cluster


def test_communities():
    np.random.seed(42)

    # Build a 100x100 image to cluster
    from sklearn.feature_extraction.image import grid_to_graph
    adj = grid_to_graph(*(100, 100))
    edges = np.where(adj.todense())
    edges = np.array([edges[0], edges[1]]).T
    edges = edges[
        edges[:, 0] < edges[:, 1]]  # Do not keep redundant edges [i.e. if (i,j) is in edges, (j,i) should not be.]

    n_samples = 3
    n_dims = 3

    # Setup a piecewise constant ground truth
    truth = np.ones((100, 100, n_dims)) / n_dims
    truth_assignments = np.zeros(truth.shape[:-1], dtype=int)
    idx = 1

    truth[25:40, 25:50] = np.random.dirichlet(np.ones(n_dims))
    truth_assignments[25:40, 25:50] = idx
    idx += 1

    truth[5:15, 60:75] = np.random.dirichlet(np.ones(n_dims))
    truth_assignments[5:15, 60:75] = idx
    idx += 1

    truth[80:95, 80:90] = np.random.dirichlet(np.ones(n_dims))
    truth_assignments[80:95, 80:90] = idx
    idx += 1

    # Sample some data
    flat_truth = truth.reshape(-1, n_dims)
    posteriors = np.array([np.random.normal(flat_truth, 0.15) for _ in range(n_samples)])

    # Cluster the posterior samples
    clusters, assignments, scores = communities_from_posteriors(posteriors, edges)
    n_clusters_selected = len(np.unique(assignments))

    # Align to the true assignments and reshape
    if n_clusters_selected < 10:
        perm = exhaustive_best_permutation(assignments, truth_assignments.reshape(-1), clusters.shape[0])
    else:
        perm = greedy_best_permutation(assignments, truth_assignments.reshape(-1), clusters.shape[0])
    assignments = swap_labels(assignments, perm)
    assignments = assignments.reshape(truth.shape[:2])

    print(f'n_clusters chosen is {n_clusters_selected}')
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    n_colors = np.unique(truth_assignments).shape[0]
    colors = sns.color_palette('colorblind', n_colors=n_colors)
    random.shuffle(colors)
    cmap = matplotlib.colors.ListedColormap(colors)
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    axarr[0].imshow(posteriors[0].reshape(truth.shape), interpolation='none')
    axarr[1].imshow(truth_assignments, cmap=cmap, vmin=0, vmax=n_colors, interpolation='none')
    axarr[2].imshow(assignments, cmap=cmap, vmin=0, vmax=n_colors, interpolation='none')
    plt.axis('off')
    plt.savefig('plots/communities.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import sys
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    from bayestme_plot import plot_result

    np.random.seed(42)

    experiment_path = sys.argv[1]
    posteriors_path = os.path.join(experiment_path, 'posteriors')
    plots_path = os.path.join(experiment_path, 'plots')
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    pos = np.load(os.path.join(experiment_path, 'pos.npy'))
    edges = np.load(os.path.join(experiment_path, 'edges.npy'))
    probs = np.load(os.path.join(posteriors_path, 'probs.npy'))[::10]

    mask = edges[:, 0] > edges[:, 1]
    edges[mask] = np.concatenate([edges[mask, 1:2], edges[mask, 0:1]], axis=1)
    print(probs.min(), probs.max())

    # Add the K=10 nearest-neighbors to the connectivity list to handle tissue gaps
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(pos.T)
    distances, indices = nbrs.kneighbors(pos.T)
    neighbors = set()
    for i, row in enumerate(indices[:, 1:]):
        for j in row:
            neighbors.add((i, j))
            neighbors.add((j, i))
    neighbors = np.array(list(neighbors))
    neighbors = neighbors[neighbors[:, 0] < neighbors[:, 1]]
    edges = np.concatenate([edges, neighbors], axis=0)

    # Cluster to find cellular communities
    print('Clustering posterior draws')
    clusters, assignments, scores = communities_from_posteriors(probs, edges, min_clusters=1, max_clusters=50,
                                                                cluster_score=gaussian_aicc_bic_mixture)
    print(f'Chose {clusters.shape[0]} clusters.')

    fig, axarr = plt.subplots(1, 1, figsize=(10, 10))

    # Get the different colors for the clusters
    n_colors = np.unique(assignments).shape[0]
    colors = sns.color_palette('colorblind', n_colors=n_colors)
    random.shuffle(colors)
    cmap = matplotlib.colors.ListedColormap(colors)

    # plot_result(axarr[0], probs.mean(axis=0)[...,0], pos, v_max=None, v_min=None, c_map=None)
    plot_result(axarr, assignments, pos, v_max=None, v_min=None, c_map=cmap, boost=True)
    axarr.invert_xaxis()
    # axarr[1].invert_xaxis()
    # axarr[0].invert_yaxis()
    # axarr[1].invert_yaxis()
    plt.savefig(os.path.join(plots_path, 'communities.pdf'), bbox_inches='tight')
    plt.close()

    plt.plot(np.arange(1, scores.shape[0] + 1), scores, lw=2, color='blue')
    plt.scatter(np.argmin(scores) + 1, scores[np.argmin(scores)], color='orange')
    plt.xlabel('Number of clusters')
    plt.ylabel('Information criterion')
    plt.savefig(os.path.join(plots_path, 'communities-bic.pdf'), bbox_inches='tight')
    plt.close()
