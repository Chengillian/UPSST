import os,csv,re,sys
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
import random, torch
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


BASE_PATH = Path(r'C:\cgy\datasets\SEDR_analyses-master\data')
output_path = Path(r'C:\cgy\all_methods\spagcn_results\\')
sample_name='Dataset1_LiuLongQi_MouseOlfactoryBulb'


dir_input = Path(f'{BASE_PATH}/{sample_name}/')
dir_output = Path(f'{output_path}/{sample_name}/')
dir_output.mkdir(parents=True, exist_ok=True)

##### read data
# print('test')
# counts_file = os.path.join(dir_input,'RNA_counts.tsv')
# coor_file = os.path.join(dir_input,'position.tsv')
# counts = pd.read_csv(counts_file, sep='\t', index_col=0)
# print('test')
# coor_df = pd.read_csv(coor_file, sep='\t')
# print('test')
# print(counts.shape, coor_df.shape)

# counts.columns = ['Spot_'+str(x) for x in counts.columns]
# coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
# coor_df = coor_df.loc[:, ['x','y']]
# adata = sc.AnnData(counts.T)
# coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
# adata.obsm["spatial"] = coor_df.to_numpy()

# print(adata)
# adata.write_h5ad(f"{BASE_PATH}/ssmob.h5ad")

# read h5ad 用上面注释的方法生成

adata = sc.read_h5ad(f'{dir_input}/ssmob.h5ad')
print(adata)
adata.var_names_make_unique()
x_array=[]
y_array=[]
for cor in adata.obsm['spatial']:
    x_array.append(cor[0])
    y_array.append(cor[1])
# print(x_array)
# print(y_array)

# Calculate adjacent matrix
b=49
a=1
adj=spg.calculate_adj_matrix(x=x_array,y=y_array, beta=b, alpha=a, histology=False)
np.savetxt(f'{dir_output}/adj.csv', adj, delimiter=',')
##### Spatial domain detection using SpaGCN
spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
spg.prefilter_specialgenes(adata)
#Normalize and take log for UMI
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
### 4.2 Set hyper-parameters
p=0.5 
spg.test_l(adj,[1, 10, 100, 500, 1000])
l=spg.find_l(p=p,adj=adj,start=1, end=2000,sep=1, tol=0.01)

n_clusters=16

r_seed=t_seed=n_seed=100
res=spg.search_res(adata, adj, l, n_clusters, start=1.0, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, 
                    t_seed=t_seed, n_seed=n_seed)
### 4.3 Run SpaGCN
clf=spg.SpaGCN()
clf.set_l(l)
#Set seed
random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)
#Run
clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
y_pred, prob=clf.predict()
adata.obs["pred"]= y_pred
adata.obs["pred"]=adata.obs["pred"].astype('category')
#Do cluster refinement(optional)
adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
adata.obs["refined_pred"]=refined_pred
adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
#Save results
# adata.write_h5ad(f"{dir_output}/results.h5ad")
# adata.obs.to_csv(f'{dir_output}/metadata.tsv', sep='\t')

#Set colors used
# adata=sc.read(f"{dir_output}/results.h5ad")
plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
#Plot spatial domains
domains="pred"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(f"{dir_output}/pred.png", dpi=300)
plt.close()
#Plot refined spatial domains
domains="refined_pred"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(f"{dir_output}/refined_pred.png", dpi=300)
plt.close()

# df_meta = pd.read_csv(f'{dir_input}/metadata.tsv', sep='\t')
# df_meta['SpaGCN'] = adata.obs["refined_pred"].tolist()
# df_meta.to_csv(f'{dir_output}/metadata.tsv', sep='\t', index=False)
# df_meta = df_meta[~pd.isnull(df_meta['fine_annot_type'])]
# ARI = metrics.adjusted_rand_score(df_meta['fine_annot_type'], df_meta['SpaGCN'])
# print('===== Project: {} ARI score: {:.3f}'.format(sample_name, ARI))

# SC AND DB
sc = metrics.silhouette_score(clf.embed,adata.obs["refined_pred"].tolist())
print(f'sc is {sc}')

db = metrics.davies_bouldin_score(clf.embed,adata.obs["refined_pred"].tolist())

print(f'db is {db}')