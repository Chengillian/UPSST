
import os,sys
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
sys.path.append("C:\cgy\codding\SEDR-master")
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file
import anndata
from src.SEDR_train import SEDR_Train
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)



# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

params = parser.parse_args()
params.device = device

# ################ Path setting
data_root = r'C:\cgy\datasets\10X_data'
data_name = 'V1_Mouse_Brain_Sagittal_Posterior'
# set saving result path
save_root = r'C:\cgy\all_methods\sedr_results\\'

if not os.path.exists(save_root):
    os.makedirs(save_root)


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

print(f'===== Project {data_name}')
file_fold = os.path.join(data_root,data_name)
# ################## Load data
adata_h5 = load_ST_file(file_fold=file_fold)
adata_h5.var_names_make_unique()
adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)
graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], params)
params.save_path = mk_dir(f'{save_root}/{data_name}/SEDR')
params.cell_num = adata_h5.shape[0]
print('==== Graph Construction Finished')
# ################## Model training
sedr_net = SEDR_Train(adata_X, graph_dict, params)
if params.using_dec:
    sedr_net.train_with_dec()
else:
    sedr_net.train_without_dec()
sedr_feat, _, _, _ = sedr_net.process()
np.savez(f'{params.save_path}/SEDR_result.npz', sedr_feat=sedr_feat, params=params)
# ################## Result plot
adata_sedr = anndata.AnnData(sedr_feat)
adata_sedr.uns['spatial'] = adata_h5.uns['spatial']
adata_sedr.obsm['spatial'] = adata_h5.obsm['spatial']
sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
# sc.tl.umap(adata_sedr)

n_clusters= 16

eval_resolution = res_search_fixed_clus(adata_sedr, n_clusters)
sc.tl.leiden(adata_sedr, key_added="SEDR_leiden", resolution=eval_resolution)
sc.pl.spatial(adata_sedr, img_key="hires", color=['SEDR_leiden'], show=False)
plt.savefig(f'{params.save_path}/SEDR_leiden_plot.jpg', bbox_inches='tight', dpi=150)


# df_meta = pd.read_csv(f'{data_root}/{data_name}/metadata.tsv', sep='\t')
# df_meta['SEDR'] = adata_sedr.obs['SEDR_leiden'].tolist()
# df_meta.to_csv(f'{params.save_path}/metadata.tsv', sep='\t', index=False)
# # #################### evaluation
# # ---------- Load manually annotation ---------------
# df_meta = df_meta[~pd.isnull(df_meta['fine_annot_type'])]
# ARI = metrics.adjusted_rand_score(df_meta['fine_annot_type'], df_meta['SEDR'])
# print('===== Project: {} ARI score: {:.3f}'.format(data_name, ARI))

# SC AND DB

refined_pred_data = []
for index in range(len(adata_sedr.obs['SEDR_leiden'])):
    refined_pred_data.append([index, int(adata_sedr.obs['SEDR_leiden'][index])])
np.savetxt(os.path.join(params.save_path, f'refined_pred_types.txt'), np.array(refined_pred_data), fmt='%3d', delimiter='\t')


sc = metrics.silhouette_score(adata_X,adata_sedr.obs['SEDR_leiden'].tolist())
print(f'sc is {sc}')

db = metrics.davies_bouldin_score(adata_X,adata_sedr.obs['SEDR_leiden'].tolist())

print(f'db is {db}')