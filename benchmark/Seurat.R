# 安装代码，执行一次即可
# https://satijalab.org/seurat/articles/install.html
chooseCRANmirror(ind=1)
# remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)
# install.packages("hdf5r")
# BiocManager::install("glmGamPoi") # https://github.com/const-ae/glmGamPoi

# 执行代码
# https://www.jianshu.com/p/137376261b13
library("Seurat")
library("ggplot2")
library("cowplot")
library("dplyr")
library("hdf5r")
##读取矩阵文件
name='Posterior1'
expr <- "C:\\cgy\\datasets\\10X_data\\V1_Mouse_Brain_Sagittal_Posterior\\V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5"
expr.mydata <- Seurat::Read10X_h5(filename =  expr )
mydata <- Seurat::CreateSeuratObject(counts = expr.mydata, project = 'Posterior1', assay = 'Spatial')
mydata$slice <- 1
mydata$region <- 'Posterior1' #命名
#读取镜像文件
imgpath <- "C:\\cgy\\datasets\\10X_data\\V1_Mouse_Brain_Sagittal_Posterior\\spatial"
img <- Seurat::Read10X_Image(image.dir = imgpath)
Seurat::DefaultAssay(object = img) <- 'Spatial'
img <- img[colnames(x = mydata)]
mydata[['image']] <- img
mydata  #查看数据


# ##UMI统计画图
# plot1 <- VlnPlot(mydata, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
# plot2 <- SpatialFeaturePlot(mydata, features = "nCount_Spatial") + theme(legend.position = "right")
# grid <- plot_grid(plot1, plot2)
# # 保存图像为 PNG 文件
# ggsave("./seurat/seurat_UMI.png", grid, dpi = 300)

# ##gene数目统计画图
# plot1 <- VlnPlot(mydata, features = "nFeature_Spatial", pt.size = 0.1) + NoLegend()
# plot2 <- SpatialFeaturePlot(mydata, features = "nFeature_Spatial") + theme(legend.position = "right")
# grid <- plot_grid(plot1, plot2)
# # 保存图像为 PNG 文件
# ggsave("./seurat/seurat_gene.png", grid, dpi = 300)


# #线粒体统计
# mydata[["percent.mt"]] <- PercentageFeatureSet(mydata, pattern = "^mt[-]")
# plot1 <- VlnPlot(mydata, features = "percent.mt", pt.size = 0.1) + NoLegend()
# plot2 <- SpatialFeaturePlot(mydata, features = "percent.mt") + theme(legend.position = "right")
# grid <-plot_grid(plot1, plot2)
# # 保存图像为 PNG 文件
# ggsave("./seurat/seurat_mt.png", grid, dpi = 300)

# mydata2 <- subset(mydata, subset = nFeature_Spatial > 200 & nFeature_Spatial <7500 & nCount_Spatial > 1000 & nCount_Spatial < 60000 & percent.mt < 25)
# mydata2
# plot1 <- VlnPlot(mydata2, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
# plot2 <- SpatialFeaturePlot(mydata2, features = "nCount_Spatial") + theme(legend.position = "right")
# grid <- plot_grid(plot1, plot2)
# ggsave("./seurat/seurat_filter_UMI.png", grid, dpi = 300)

mydata <- SCTransform(mydata, assay = "Spatial", verbose = FALSE)

mydata <- RunPCA(mydata, assay = "SCT", verbose = FALSE)
mydata <- FindNeighbors(mydata, reduction = "pca", dims = 1:30)
mydata <- FindClusters(mydata, verbose = FALSE)
mydata <- RunUMAP(mydata, reduction = "pca", dims = 1:30)
umap_plot <- DimPlot(mydata, reduction = "umap")

# 将聚类标签保存为文本文件
write.table(Idents(object = mydata, slot = "Spatial", ident.1 = "cluster"), file = "cluster_labels.txt", sep = "\t", quote = FALSE, col.names = FALSE, row.names = TRUE)

# 保存图像为 PNG 文件
ggsave("umap_plot.png", umap_plot)

# # 将聚类标签添加到原始图像中
# mydata <- AddCellTypes(mydata, ident.1 = "cluster")
# spatial_plot_with_cluster <- SpatialPlot(mydata, features = "SCT", cols = c("blue", "red"), pt.size = 0.5)

# # 将 UMAP 图和带聚类标签的图合并到一起
# combined_plot <- plot_grid(umap_plot, spatial_plot_with_cluster)

# 保存图像为 PNG 文件
ggsave("combined_plot.png", combined_plot)

mydata <- RunTSNE(mydata, reduction = "pca",dims = 1:30)
tsne_plot <- DimPlot(mydata, reduction = "tsne")
# 保存图像为 PNG 文件
ggsave("tsne_plot.png", tsne_plot)