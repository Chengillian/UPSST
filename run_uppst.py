from uppst import train
import os
import random
import numpy as np
from scipy import sparse
import pickle
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn import metrics
import matplotlib

matplotlib.use('Agg')

import torch

def run(args):
    r_seed = t_seed = n_seed = 100
    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='151669',
                        help="'MERFISH' or 'V1_Breast_Cancer_Block_A_Section_1")
    parser.add_argument('--lambda_I', type=float, default=0.3)
    parser.add_argument('--data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data')
    parser.add_argument( '--result_path', type=str, default='results')
    parser.add_argument( '--DGI', type=int, default=1, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
    parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
    parser.add_argument( '--hidden', type=int, default=512, help='hidden channels in DGI')
    parser.add_argument( '--PCA', type=int, default=1, help='run PCA or not')
    parser.add_argument( '--cluster', type=int, default=1, help='run cluster or not')
    parser.add_argument( '--n_clusters', type=int, default=16, help='number of clusters in Kmeans, when ground truth label is not avalible.') #5 on MERFISH, 20 on Breast
    parser.add_argument( '--draw_map', type=int, default=1, help='run drawing map')
    parser.add_argument( '--diff_gene', type=int, default=0, help='Run differential gene expression analysis')
    parser.add_argument('--img_decoder', type=str, default='middle_result.npy')
    parser.add_argument('--with_img',type=bool,default=True,help='whether with img encoder')

    args = parser.parse_args()
    args.embedding_data_path = args.embedding_data_path +'/'+ args.data_name +'/'
    args.model_path = args.model_path +'/'+ args.data_name +'/'
    args.result_path = args.result_path +'/'+ args.data_name +'/'
    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print ('------------------------Model and Training Details--------------------------')
    print(args)
    run(args)