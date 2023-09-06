args = commandArgs(trailingOnly=TRUE)
# sample.name <- args[1]
# n_clusters <- args[2]
sample.name <- 'STARmap'
n_clusters <- 16

print(sample.name)

library(devtools)
# install_github("satijalab/seurat-data")

chooseCRANmirror(ind=1)
# install.packages("clusterSim")
# install_github("mojaveazure/seurat-disk")
install_github("zhanghao-njmu/SCP")
library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
options(bitmapType = 'cairo')
library(clusterSim)

chooseCRANmirror(ind=1)
# install.packages("anndata")
library(anndata)
library(SeuratDisk)

dir.input <- file.path('C:\\cgy\\datasets\\10X_data\\STARmap\\annotation')
dir.output <- file.path('./output/', sample.name, '/Seurat/')

print(dir.input)

if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}

library(SCP)
library(reticulate)
sc <- import("scanpy")
adata <- sc$read_h5ad("pancreas.h5ad")
sp_data <- adata_to_srt(adata)

# 读取h5ad文件
# Convert(file.path(dir.input, 'STARmap_20180505_BY3_1k.h5ad'), file.path(dir.input, 'STARmap_20180505_BY3_1k.h5seurat'))
# sp_data <- LoadH5Seurat(file.path(dir.input, 'STARmap_20180505_BY3_1k.h5seurat'))


# df_meta <- read.table(file.path(dir.input, 'metadata.tsv'), header = TRUE, sep = "\t")
# sp_data <- AddMetaData(sp_data, 
#                        metadata = df_meta$fine_annot_type,
#                        col.name = 'fine_annot_type')

### Data processing
plot1 <- VlnPlot(sp_data, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(sp_data, features = "nCount_Spatial") + theme(legend.position = "right")
wrap_plots(plot1, plot2)
ggsave(file.path(dir.output, './Seurat.QC.png'), width = 10, height=5)

# sctransform
sp_data <- SCTransform(sp_data, assay = "Spatial", verbose = FALSE)


### Dimensionality reduction, clustering, and visualization
sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:30)

for(resolution in 50:30){
  sp_data <- FindClusters(sp_data, verbose = F, resolution = resolution/100)
  if(length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters){
    break
  }
}
sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = 0.46)
sp_data <- RunUMAP(sp_data, reduction = "pca", dims = 1:30)

p1 <- DimPlot(sp_data, reduction = "umap", label = TRUE)
p2 <- SpatialDimPlot(sp_data, label = TRUE, label.size = 3)
p1 + p2
ggsave(file.path(dir.output, './Seurat.cell_cluster.png'), width=10, height=5)


##### save data
saveRDS(sp_data, file.path(dir.output, 'Seurat_final.rds'))

write.table(sp_data@reductions$pca@cell.embeddings, file = file.path(dir.output, 'seurat.PCs.tsv'), sep='\t', quote=F)

write.table(sp_data@meta.data, file = file.path(dir.output, './metadata.tsv'), sep='\t', quote=FALSE)


##### 
library(mclust)
# print(adjustedRandIndex(sp_data@meta.data$fine_annot_type, sp_data@meta.data$seurat_clusters))

library(cluster,quietly =TRUE)
cluster_labels<- as.numeric(sp_data@meta.data$seurat_clusters)
s <- silhouette(cluster_labels,dist(as.matrix(sp_data@reductions$pca@cell.embeddings)))
print('silhouette score is')
print(mean(s[, 3]))

# library(clusterSim)
# dbi <- index.DB(sp_data@reductions$pca@cell.embeddings,sp_data@meta.data$seurat_clusters)
# print('davies bouldin score is')
# print(dbi)

