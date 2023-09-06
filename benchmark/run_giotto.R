library(Giotto)
chooseCRANmirror(ind=1)
# # install.packages("stringi")
# # install.packages("devtools")

# # install.packages("usethis")
# # usethis::create_github_token()
# # usethis::edit_r_environ()
# devtools::install_github("drieslab/Giotto@master",force = TRUE)

install.packages('FactoMineR')

# 1. set working directory
results_folder = './giotto/'

# 2. set giotto python path
# set python path to your preferred python version path
# set python path to NULL if you want to automatically install (only the 1st time) and use the giotto miniconda environment
python_path = 'C:\\ProgramData\\Anaconda3\\python.exe'
if(is.null(python_path)) {
  installGiottoEnvironment()
}
# download data to working directory
# if wget is installed, set method = 'wget'
# if you run into authentication issues with wget, then add " extra = '--no-check-certificate' "
# getSpatialDataset(dataset = 'starmap_3D_cortex', directory = results_folder, method = 'wget')
## instructions allow us to automatically save all plots into a chosen results folder
instrs = createGiottoInstructions(show_plot = FALSE,
                                  save_plot = TRUE, 
                                  save_dir = results_folder,
                                  python_path = python_path)

expr_path = paste0(results_folder, "STARmap_3D_data_expression.txt")
loc_path = paste0(results_folder, "STARmap_3D_data_cell_locations.txt")

## create
STAR_test <- createGiottoObject(raw_exprs = expr_path,
                                spatial_locs = loc_path,
                                instructions = instrs)

## filter raw data
# pre-test filter parameters
filterDistributions(STAR_test, detection = 'genes',
                    save_param = list(save_name = '2_a_distribution_genes'))
filterDistributions(STAR_test, detection = 'cells',
                    save_param = list(save_name = '2_b_distribution_cells'))
# filter
STAR_test <- filterGiotto(gobject = STAR_test,
                          gene_det_in_min_cells = 20000,
                          min_det_genes_per_cell = 20)
## normalize
STAR_test <- normalizeGiotto(gobject = STAR_test, scalefactor = 10000, verbose = T)
STAR_test <- addStatistics(gobject = STAR_test)
STAR_test <- adjustGiottoMatrix(gobject = STAR_test, expression_values = c('normalized'),
                                batch_columns = NULL, covariate_columns = c('nr_genes', 'total_expr'),
                                return_gobject = TRUE,
                                update_slot = c('custom'))
## visualize
# 3D
spatPlot3D(gobject = STAR_test, point_size = 2,
           save_param = list(save_name = '2_d_spatplot_3D'))


STAR_test <- calculateHVG(gobject = STAR_test, method = 'cov_groups', 
                          zscore_threshold = 0.5, nr_expression_groups = 3,
                          save_param = list(save_name = '3_a_HVGplot', base_height = 5, base_width = 5))

# too few highly variable genes
# genes_to_use = NULL is the default and will use all genes available
STAR_test <- runPCA(gobject = STAR_test, genes_to_use = NULL, scale_unit = F,method = 'factominer')
signPCA(STAR_test,
        save_param = list(save_name = '3_b_signPCs'))


STAR_test <- runUMAP(STAR_test, dimensions_to_use = 1:8, n_components = 3, n_threads = 4)
plotUMAP_3D(gobject = STAR_test,
            save_param = list(save_name = '3_c_UMAP'))

## sNN network (default)
STAR_test <- createNearestNetwork(gobject = STAR_test, dimensions_to_use = 1:8, k = 15)

## Leiden clustering
STAR_test <- doLeidenCluster(gobject = STAR_test, resolution = 0.2, n_iterations = 100,
                             name = 'leiden_0.2')

plotUMAP_3D(gobject = STAR_test, cell_color = 'leiden_0.2',show_center_label = F,
            save_param = list(save_name = '4_a_UMAP'))
  

spatDimPlot3D(gobject = STAR_test,
              cell_color = 'leiden_0.2',
              save_param = list(save_name = '5_a_spatDimPlot'))

markers = findMarkers_one_vs_all(gobject = STAR_test,
                                 method = 'gini',
                                 expression_values = 'normalized',
                                 cluster_column = 'leiden_0.2',
                                 min_expr_gini_score = 2,
                                 min_det_gini_score = 2,
                                 min_genes = 5, rank_score = 2)
markers[, head(.SD, 2), by = 'cluster']

# violinplot
violinPlot(STAR_test, genes = unique(markers$genes), cluster_column = 'leiden_0.2',
           strip_position = "right", save_param = list(save_name = '6_a_violinplot'))


markers = findMarkers_one_vs_all(gobject = STAR_test,
                                 method = 'gini',
                                 expression_values = 'normalized',
                                 cluster_column = 'leiden_0.2',
                                 min_expr_gini_score = 2,
                                 min_det_gini_score = 2,
                                 min_genes = 5, rank_score = 2)
markers[, head(.SD, 2), by = 'cluster']

# violinplot
violinPlot(STAR_test, genes = unique(markers$genes), cluster_column = 'leiden_0.2',
           strip_position = "right", save_param = list(save_name = '6_a_violinplot'))