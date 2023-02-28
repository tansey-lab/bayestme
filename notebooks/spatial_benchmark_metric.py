"""
Adpoted from Li et al 2022
original file can be found at https://github.com/QuKunLab/SpatialBenchmarking/issues/6
"""
import os
import pandas as pd
import numpy as np
from collections.abc import Iterable
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu


def ssim(im1, im2, M=1):
    im1, im2 = im1 / im1.max(), im2 / im2.max()
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def rmse(x1, x2):
    return mean_squared_error(x1, x2, squared=False)


def mae(x1, x2):
    return np.mean(np.abs(x1 - x2))


def compare_results(gd, result_list, metric="pcc", columns=None, axis=1):
    if metric == "pcc":
        func = pearsonr
        r_ind = 0
    if metric == "mae":
        func = mae
        r_ind = None
    if metric == "jsd":
        func = jensenshannon
        r_ind = None
    if metric == "rmse":
        func = rmse
        r_ind = None
    if metric == "ssim":
        func = ssim
        r_ind = None
    if isinstance(result_list, pd.DataFrame):
        c_list = []
        if axis == 1:
            print("axis: ", 1)
            for i, c in enumerate(gd.columns):
                r = func(gd.iloc[:, i].values, np.clip(result_list.iloc[:, i], 0, 1))
                if isinstance(result_list, Iterable):
                    if r_ind is not None:
                        r = r[r_ind]
                c_list.append(r)
        else:
            print("axis: ", 0)
            for i, c in enumerate(gd.index):
                r = func(gd.iloc[i, :].values, np.clip(result_list.iloc[i, :], 0, 1))
                if isinstance(result_list, Iterable):
                    if r_ind is not None:
                        r = r[r_ind]
                c_list.append(r)
        df = pd.DataFrame(c_list, index=gd.columns, columns=columns)
    else:
        df_list = []
        for res in result_list:
            c_list = []
            if axis == 1:
                for i, c in enumerate(gd.columns):
                    r = func(gd.iloc[:, i].values, np.clip(res.iloc[:, i], 0, 1))
                    if isinstance(res, Iterable):
                        if r_ind is not None:
                            r = r[r_ind]
                    c_list.append(r)
                df_tmp = pd.DataFrame(c_list, index=gd.columns)
            else:
                for i, c in enumerate(gd.index):
                    r = func(gd.iloc[i, :].values, np.clip(res.iloc[i, :], 0, 1))
                    if isinstance(res, Iterable):
                        if r_ind is not None:
                            r = r[r_ind]
                    c_list.append(r)
                df_tmp = pd.DataFrame(c_list, index=gd.index)
            df_list.append(df_tmp)
        df = pd.concat(df_list, axis=1)
        df.columns = columns
    return df


def load_benchmark_results(data_path):
    """
    load ground truth cell type proportion
    """
    gd_results = pd.read_table(
        os.path.join(data_path, "combined_spot_clusters.txt"),
        sep="\t",
        header=0,
        index_col=0,
    )
    gd_results.columns = [
        "Astro",
        "Endo",
        "Excitatory L2/3",
        "Excitatory L4",
        "Excitatory L5",
        "Excitatory L6",
        "HPC",
        "Micro",
        "Npy",
        "Olig",
        "Other",
        "Inhibitory Pvalb",
        "Smc",
        "Inhibitory Sst",
        "Inhibitory Vip",
    ]
    gd_results = (gd_results.T / gd_results.sum(axis=1)).T
    gd_results = gd_results.drop(columns=["Other", "Npy", "HPC"])
    gd_results = gd_results.loc[:, np.unique(gd_results.columns)]
    gd_results = gd_results.fillna(0)
    sc_rna_meta = pd.read_csv(
        os.path.join(data_path, "starmap_sc_rna_celltype.tsv"),
        sep="\t",
        header=None,
        index_col=0,
    )

    """
    load benchmark results
    """
    results_path = os.path.join(data_path, "Result_STARmap")
    # Tangram
    tangram_results = pd.read_csv(
        os.path.join(results_path, "Tangram_result.txt"), index_col=0
    )
    tangram_results = tangram_results.loc[:, np.unique(tangram_results.columns)]

    # SpaOTsc
    spa_map = np.load(os.path.join(results_path, "SpaOTsc_alignment.npy"))
    spa_results = pd.DataFrame(
        np.zeros((spa_map.shape[1], len(np.unique(sc_rna_meta[1])))),
        columns=np.unique(sc_rna_meta[1]),
    )
    for i, l in enumerate(np.argmax(spa_map, axis=1)):
        spa_results.loc[l, sc_rna_meta.iloc[i, 0]] += 1

    # novoSpaRc
    novo_map = np.load(os.path.join(results_path, "novoSpaRc_alignment.npy"))
    novo_results = pd.DataFrame(
        np.zeros((novo_map.shape[1], len(np.unique(sc_rna_meta[1])))),
        columns=np.unique(sc_rna_meta[1]),
    )
    for i, l in enumerate(np.argmax(novo_map, axis=1)):
        novo_results.loc[l, sc_rna_meta.iloc[i, 0]] += 1

    # cell2location
    cell2loc_results = pd.read_csv(
        os.path.join(results_path, "Cell2location_result.txt"), index_col=0
    )
    cell2loc_results.index = np.arange(len(cell2loc_results))
    cell2loc_results.columns = [
        c.split("q05cell_abundance_w_sf_")[1] for c in cell2loc_results.columns
    ]
    cell2loc_results = cell2loc_results.loc[:, np.unique(cell2loc_results.columns)]

    # RCTD
    RCTD_results = pd.read_csv(
        os.path.join(results_path, "RCTD_result.txt"), index_col=0
    )
    RCTD_results.index = np.arange(len(RCTD_results))
    RCTD_results = RCTD_results.loc[:, np.unique(RCTD_results.columns)]
    RCTD_results.columns = tangram_results.columns

    # SpatialDWLS
    spatialdwls_results = pd.read_csv(
        os.path.join(results_path, "SpatialDWLS_result.txt"), index_col=0
    )
    spatialdwls_results.index = np.arange(len(spatialdwls_results))
    spatialdwls_results = spatialdwls_results.iloc[:, 1:]
    spatialdwls_results = spatialdwls_results.loc[
        :, np.unique(spatialdwls_results.columns)
    ]

    # Stereoscope
    stereo_results = pd.read_csv(
        os.path.join(results_path, "Stereoscope_result.txt"), index_col=0
    )
    stereo_results.index = np.arange(len(stereo_results))
    stereo_results = stereo_results.loc[:, np.unique(stereo_results.columns)]
    stereo_results.columns = tangram_results.columns
    stereo_results = (stereo_results.T / stereo_results.sum(axis=1)).T
    stereo_results = stereo_results.fillna(0)

    # DestVI
    destvi_results = pd.read_csv(
        os.path.join(results_path, "DestVI_result.txt"), index_col=0
    )
    destvi_results.index = np.arange(len(destvi_results))
    destvi_results = destvi_results.loc[:, np.unique(destvi_results.columns)]
    destvi_results.columns = tangram_results.columns
    destvi_results = (destvi_results.T / destvi_results.sum(axis=1)).T
    destvi_results = destvi_results.fillna(0)

    # SPOTlight
    spotlight_results = pd.read_csv(
        os.path.join(results_path, "SPOTlight_result.txt"), index_col=0
    )
    spotlight_results.index = np.arange(len(spotlight_results))
    spotlight_results = spotlight_results.loc[:, np.unique(spotlight_results.columns)]
    spotlight_results.columns = tangram_results.columns
    spotlight_results = (spotlight_results.T / spotlight_results.sum(axis=1)).T
    spotlight_results = spotlight_results.fillna(0)

    # STRIDE
    stride_results = pd.read_csv(
        os.path.join(results_path, "STRIDE_result.txt"), index_col=0, sep="\t"
    )
    stride_results.index = np.arange(len(stride_results))
    stride_results = stride_results.loc[:, np.unique(stride_results.columns)]
    stride_results.columns = tangram_results.columns
    stride_results = (stride_results.T / stride_results.sum(axis=1)).T
    stride_results = stride_results.fillna(0)

    # Seurat
    seurat_results = pd.read_csv(
        os.path.join(results_path, "Seurat_result.txt"), index_col=0
    )
    seurat_results = seurat_results.iloc[:, 1:-1]
    seurat_results = seurat_results.loc[:, np.unique(seurat_results.columns)]
    seurat_results.columns = tangram_results.columns

    # DSTG
    dstg_results = pd.read_csv(
        os.path.join(results_path, "DSTG_result.txt"), sep=",", header=None
    )
    label = pd.read_csv(os.path.join(results_path, "Label.csv"), sep=",")
    dstg_results.columns = label.columns
    dstg_results = dstg_results.loc[:, np.unique(dstg_results.columns)]
    dstg_results.columns = tangram_results.columns

    # clean up benchmark results
    tangram_results = (tangram_results.T / tangram_results.sum(axis=1)).T
    seurat_results = (seurat_results.T / seurat_results.sum(axis=1)).T
    cell2loc_results = (cell2loc_results.T / cell2loc_results.sum(axis=1)).T
    novo_results = (novo_results.T / novo_results.sum(axis=1)).T
    spa_results = (spa_results.T / spa_results.sum(axis=1)).T

    tangram_results = tangram_results.fillna(0)
    seurat_results = seurat_results.fillna(0)
    cell2loc_results = cell2loc_results.fillna(0)
    novo_results = novo_results.fillna(0)
    spa_results = spa_results.fillna(0)

    tangram_results = tangram_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    seurat_results = seurat_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    RCTD_results = RCTD_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    cell2loc_results = cell2loc_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    stereo_results = stereo_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    destvi_results = destvi_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    spotlight_results = spotlight_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    spatialdwls_results = spatialdwls_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    dstg_results = dstg_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    stride_results = stride_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    novo_results = novo_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )
    spa_results = spa_results.drop(
        columns=["Other", "Neuron Other", "Inhibitory Other"]
    )

    benchmark_results = [
        RCTD_results,
        cell2loc_results,
        tangram_results,
        seurat_results,
        stereo_results,
        spotlight_results,
        spatialdwls_results,
        destvi_results,
        dstg_results,
        stride_results,
        spa_results,
        novo_results,
    ]
    return gd_results, benchmark_results


def get_score(Result):
    tools_num = Result.shape[0]
    Tools_score = []
    methods = list(Result.index)
    score_col = []
    list_up = list(range(1, Result.shape[1] + 1))
    list_down = list(range(Result.shape[1], 0, -1))

    for method in methods:
        if method == "PCC" or method == "SSIM":
            Tools_score.append(
                pd.Series(
                    list_down,
                    index=Result.loc[method, :].sort_values(ascending=False).index,
                )
            )

        if method == "JS" or method == "RMSE":
            Tools_score.append(
                pd.Series(
                    list_up,
                    index=Result.loc[method, :].sort_values(ascending=False).index,
                )
            )
        score_col.append(method)

    score = pd.concat([m for m in Tools_score], axis=1)
    score.columns = score_col
    score = score / Result.shape[1]
    return score


def make_score(dataset, Tools):
    prefix = dataset
    Tools_data = [x for x in range(len(Tools))]
    for i in range(len(Tools)):
        File = prefix + "_" + Tools[i] + "_Metrics.txt"
        if os.path.isfile(File):
            Tools_data[i] = pd.read_table(File, sep="\t", index_col=0, header=0)
            Tools_data[i] = Tools_data[i].mean()
            Tools_data[i]["Tool"] = Tools[i]
        else:
            print(File)
            Tools_data[i] = pd.DataFrame(
                [-1, -1, 1, 1], columns=["Genes"], index=["PCC", "SSIM", "RMSE", "JS"]
            ).T
            Tools_data[i] = Tools_data[i].mean()
            Tools_data[i]["Tool"] = Tools[i]
    Result = pd.concat([m for m in Tools_data], axis=1)
    Result.columns = Result.loc[["Tool"], :].values.flatten()
    Result.drop("Tool", axis=0, inplace=True)

    score = get_score(Result)
    return score
