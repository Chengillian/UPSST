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


BASE_PATH = Path(r'C:\cgy\datasets\10X_data\STARmap\annotation')
output_path = Path(r'C:\cgy\all_methods\spagcn_results\\')
sample_name='STARmap'


dir_input = Path(f'{BASE_PATH}/STARmap_20180505_BY3_1k.h5ad')
dir_output = Path(f'{output_path}/{sample_name}/')
dir_output.mkdir(parents=True, exist_ok=True)

##### read data
adata = sc.read(f'{dir_input}')
adata.var_names_make_unique()
print(adata)

x_array=adata.obs["X"].tolist()

y_array=adata.obs["Y"].tolist()

#Calculate adjacent matrix
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
l=spg.find_l(p=p,adj=adj,start=10, end=500,sep=1, tol=0.01)

n_clusters=7

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
ax=sc.pl.scatter(adata,alpha=1,x="X",y="Y",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(f"{dir_output}/pred.png", dpi=300)
plt.close()
#Plot refined spatial domains
domains="refined_pred"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
ax=sc.pl.scatter(adata,alpha=1,x="X",y="Y",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(f"{dir_output}/refined_pred.png", dpi=300)
plt.close()

pred_data = []
for index in range(len(adata.obs["pred"])):
    pred_data.append([index, int(adata.obs["pred"][index])])
np.savetxt(os.path.join(dir_output, f'pred_types.txt'), np.array(pred_data), fmt='%3d', delimiter='\t')

refined_pred_data = []
for index in range(len(adata.obs["refined_pred"])):
    refined_pred_data.append([index, int(adata.obs["refined_pred"][index])])
np.savetxt(os.path.join(dir_output, f'refined_pred_types.txt'), np.array(refined_pred_data), fmt='%3d', delimiter='\t')

ARI = metrics.adjusted_rand_score(adata.obs['label'], adata.obs["refined_pred"])
print('===== Project: {} ARI score: {:.3f}'.format(sample_name, ARI))

