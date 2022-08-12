library(dplyr)
library(Seurat)
library(patchwork)
library(findPC)
library(optparse)
library(ggplot2)


option_list = list(
  make_option("--dataDir", action="store", type='character',
              help="Directory containing cellranger output."),
  make_option("--outputDir", action="store", type='character',
              help="Output directory")
)

opt = parse_args(OptionParser(option_list=option_list))

dir.create(file.path(opt$outputDir))
setwd(file.path(opt$outputDir))

sc_sample.data <- Read10X(data.dir = opt$dataDir)
sc_sample <- CreateSeuratObject(counts = sc_sample.data, project = "sc_sample", min.cells = 3, min.features = 200)

sc_sample[["percent.mt"]] <- PercentageFeatureSet(sc_sample, pattern = "^MT-")
VlnPlot(sc_sample, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
ggsave("violin_plot.pdf", width=14, height=7)

plot1 <- FeatureScatter(sc_sample, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(sc_sample, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2
ggsave("feature_scatter_plot.pdf", width=14, height=7)

sc_sample <- subset(sc_sample, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
sc_sample <- NormalizeData(sc_sample)
sc_sample <- FindVariableFeatures(sc_sample, selection.method = "vst", nfeatures = 2000)
# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(sc_sample), 10)
# plot variable features with and without labels
plot1 <- VariableFeaturePlot(sc_sample)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

ggsave("variable_features_plot.pdf", width=14, height=7)
all.genes <- rownames(sc_sample)
sc_sample <- ScaleData(sc_sample, features = all.genes)
sc_sample <- RunPCA(sc_sample, features = VariableFeatures(object = sc_sample))

total_components = length(Stdev(object = sc_sample, reduction = "pca"))

VizDimLoadings(sc_sample, dims = 1:min(10, total_components), reduction = "pca")

ggsave("dim_loadings_plot.pdf", width=14, height=14)

sc_sample <- JackStraw(sc_sample, num.replicate = 100)
sc_sample <- ScoreJackStraw(sc_sample, dims = 1:min(20, total_components))

ElbowPlot(sc_sample)
ggsave("elbow_plot.pdf", width=14, height=7)


JackStrawPlot(sc_sample, dims = 1:min(20, total_components))

ggsave("jack_straw_plot.pdf", width=14, height=7)

n_principal_components = findPC(
    sdev = Stdev(object = sc_sample, reduction = "pca"),
    number = c(total_components),
    method = 'all',
    aggregate = 'voting', figure=T)

print(paste("Using", n_principal_components, "principal components"))

ggsave("pc_selection_plot.pdf", width=14, height=7)

sc_sample <- FindNeighbors(sc_sample, dims = 1:n_principal_components)
sc_sample <- FindClusters(sc_sample, resolution = 0.5)

sc_sample <- RunUMAP(sc_sample, dims = 1:n_principal_components)
DimPlot(sc_sample, reduction = "umap")

ggsave("umap_plot.pdf", width=14, height=14)

sc_sample.markers <- FindAllMarkers(sc_sample, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
sc_sample.markers %>%
    group_by(cluster) %>%
    slice_max(n = 2, order_by = avg_log2FC)

top10 = sc_sample.markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC)
DoHeatmap(sc_sample, features = top10$gene) + NoLegend()
ggsave("heatmap_plot.pdf", width=14, height=14)

avg_exp = AverageExpression(object=sc_sample, slot="counts")$RNA

write.csv(avg_exp, "average_expression.csv")

clusters = sc_sample[[c("seurat_clusters")]]
write.csv(clusters, "umi_to_cluster.csv")
