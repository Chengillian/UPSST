import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn import metrics
import scanpy as sc
from sklearn.cluster import KMeans


from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)

        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)

        return x

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)

class UPPST(nn.Module):
    def __init__(self,in_channels,
                 hidden_channels,
                 n_clusters,
                 alpha=0.2):
        super(UPPST, self).__init__()
        self.alpha=alpha
        self.dgi = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder= Encoder(in_channels=in_channels,hidden_channels=hidden_channels),
            summary = lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )
        self.conv = GCNConv(in_channels=hidden_channels, out_channels=in_channels)
        self.mu = Parameter(torch.Tensor(n_clusters, in_channels))

    def forward(self,data):
        pos_z, neg_z, summary = self.dgi(data=data)

        # https://blog.csdn.net/qq_33309098/article/details/122305551
        x=self.conv(pos_z,data.edge_index,data.edge_weight)

        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)

        return pos_z, neg_z, summary, x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def uppst_loss(self,pos_z,neg_z,summary,p,q):
        dgi_loss = self.dgi.loss(pos_z,neg_z,summary)
        gcn_loss = self.loss_function(p,q)
        loss = dgi_loss+gcn_loss
        print(f'total_loss:{loss}, dgi_loss:{dgi_loss}, gcn_loss{gcn_loss}')
        return  loss



def train_uppst(args,data_loader,in_channels,hidden_channels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uppst_model = UPPST(
        in_channels=in_channels,
        hidden_channels=hidden_channels
    )


class train():
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 n_clusters,
                 data,
                 update_interval = 3,
                 lr =1e-6,
                 weight_decay = 1e-4,
                 pre_epochs=100,
                 num_epochs = 5000,
                 use_gpu=True):
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.update_interval = update_interval
        self.pre_epochs =pre_epochs
        self.num_epochs = num_epochs
        self.model = UPPST(in_channels,hidden_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.data = data.to(self.device)
        self.in_channels = in_channels

        self.model.mu = Parameter(torch.Tensor(n_clusters, self.in_channels).to(self.device))


    def pretrain(self):
        for epoch in self.pre_epochs:
            self.model.train()
            self.optimizer.zero_grad()
            pos_z, neg_z, summary, x, q = self.model(
                self.data
            )
            loss = self.model.uppst_loss(pos_z, neg_z, summary, x, q)
            print(f'pretrain epoch {epoch}, loss: {loss}')
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def process(self):
        self.model.eval()
        pos_z, neg_z, summary, x, q = self.model(
            self.data
        )
        return  pos_z, neg_z, summary, x, q

    def save_model(
        self,
        save_model_file
        ):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(
        self,
        save_model_file
        ):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def convert(self,f):
        return f.cpu().detach().numpy()

    def fit(self,
            n_clusters,
            cluster_type='kmeans',
            res=0.5):
        self.pretrain()
        pos_z, neg_z, summary, x, q  =self.process()
        if cluster_type =='kmeans':
            print(f"Initializing cluster centers with kmeans, n_clusters is {n_clusters}")

            cluster_method = KMeans(n_clusters=n_clusters, n_init=n_clusters * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(self.convert(x)))
            self.model.mu.data.copy_(torch.Tensor(cluster_method.cluster_centers_).to(self.device))

        # elif cluster_type =='louvain':
        #     # 暂时没有实现
        #     print(f"Initializing cluster centers with louvain, n_clusters is {n_clusters}, resolution is {res} ")
        #     adata = sc.AnnData(self.convert(x))
        #     sc.pp.neighbors(adata, n_neighbors=n_clusters)
        #     sc.tl.louvain(adata, resolution=res)
        #     y_pred_last = adata.obs['louvain'].astype(int).to_numpy()
        #     n_clusters = len(np.unique(y_pred_last))
        #     print(f'louvain n_clusters is {n_clusters}')

        for epoch in range(self.num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            if epoch%self.update_interval ==0:
                pos_z, neg_z, summary, x, q = self.process()
                p  = self.model.target_distribution(q).data

            pos_z, neg_z, summary, x, q=self.model(self.data)
            loss = self.model.uppst_loss(
                pos_z=pos_z,
                neg_z=neg_z,
                summary=summary,
                p=p,
                q=q
            )
            print(f'Train epoch is {epoch} loss is {loss}')
            loss.backward()
            self.optimizer.step()