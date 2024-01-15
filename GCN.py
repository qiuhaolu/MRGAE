import numpy as np
import torch.nn as nn
from datetime import datetime
import math
import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.nn.inits import reset
from torch_geometric.utils import to_undirected, negative_sampling
import random

EPS = 1e-15

 
class MRGAE(GAE):
    def __init__(self, encoder, discriminator, decoder=None):

        super(MRGAE, self).__init__(encoder, decoder)

        self.discriminator = discriminator

        reset(self.discriminator)

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = math.floor(val_ratio * row.size(0))
        n_t = math.floor(test_ratio * row.size(0))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data




    def reset_parameters(self):
        super(MRGAE, self).reset_parameters()
        reset(self.discriminator)
        #reset(self.discriminator2)

    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def reg_loss2(self, z):
        r"""Computes the regularization loss of the encoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator2(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def discriminator_loss(self, z):
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss

    def discriminator_loss_2views(self, z1, z2):
        fake1 = torch.sigmoid(self.discriminator(z1.detach()))
        fake2 = torch.sigmoid(self.discriminator(z2.detach()))
        fake_loss1 = -torch.log(fake1 + EPS).mean()
        fake_loss2 = -torch.log(1 - fake2 + EPS).mean()
        return fake_loss1 + fake_loss2
    
    def recon_loss_2views(self, z, pos_edge_index, z2, pos_edge_index2):
        #view 1
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        #view 2
        pos_loss2 = -torch.log(
            self.decoder(z2, pos_edge_index2, sigmoid=True) + EPS).mean()

        neg_edge_index2 = negative_sampling(pos_edge_index2, z.size(0))
        neg_loss2 = -torch.log(
            1 - self.decoder(z2, neg_edge_index2, sigmoid=True) + EPS).mean()
        return (pos_loss + neg_loss) + 1 * (pos_loss2 + neg_loss2)   - 1*torch.log(torch.sigmoid((z[pos_edge_index[0]] * z2[pos_edge_index[1]]).sum(dim=1))+ EPS).mean()
  






randomseed = 3
torch.manual_seed(randomseed)


class Discriminator(torch.nn.Module):
    def __init__(self, n_input):
        super(Discriminator, self).__init__()
        num_h1 = 128
        num_h2 = 128
        #num_h3 = 64
        #LeakyReLU
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(n_input,num_h1), torch.nn.LeakyReLU())
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(num_h1,num_h2), torch.nn.LeakyReLU())
        #self.fc4 = torch.nn.Sequential(torch.nn.Linear(num_h2,num_h3), torch.nn.ReLU())
        self.fc3 = torch.nn.Sequential(torch.nn.Linear(num_h2,1))

    def forward(self, z):
        z=self.fc1(z)
        z=self.fc2(z)
        #z=self.fc4(z)
        out=self.fc3(z)

        
        return out




channels = 32 # embedding dim



#filename = 'cora/cora.cites'
#filename_content = 'cora/cora.content'
#filename_v2 = 'cora/cora.id_view2_0.5'
#filename_v2 = 'cora/cora.id_view2_0.7'


filename = 'citeseer/citeseer.cites'
filename_content = 'citeseer/citeseer.content'
#filename_v2 = 'citeseer/citeseer.id_view2_0.3'
#filename_v2 = 'citeseer/citeseer.id_view2_0.5'
filename_v2 = 'citeseer/citeseer.id_view2_0.7'


#filename = 'pubmed/pubmed.cites'
#filename_content = 'pubmed/pubmed.content'
#filename_v2 = 'pubmed/pubmed.id_view2_0.7'



#filename = 'WebKB/wisconsin.cites'
#filename_content = 'WebKB/wisconsin.content'
#filename_v2 = 'WebKB/wisconsin.id_view2_0.5'



#filename = 'WebKB/webkb.cites'
#filename_content = 'WebKB/webkb.content'
#filename_v2 = 'WebKB/webkb.id_view2_0.5'



#filename = 'TerrorAttack/terrorist_attack_loc_org.edges'
#filename = 'TerrorAttack/terrorist_attack_loc.edges'
#filename_content = 'TerrorAttack/terrorist_attack.nodes'
#filename_v2 = 'TerrorAttack/terrorist_attack.id_view2_0.8'

#filename = 'all_closure_edges_icd9.tsv'
#filename_content = 'all_closure_nodes_icd9.tsv'
#filename_v2 = 'all_closure_edges_icd9.tsv'




#get node2id from content, which could be different from cites
node2id = {}
f1 = open(filename_content)
cnt = 0
for line in f1:
    values = line.split('\t')
    #print(values)
    l = values[0].strip()
    #r = values[1].strip()
    if l not in node2id:
        node2id[l] = cnt
        cnt += 1
    #if r not in node2id:
    #    node2id[r] = cnt
    #    cnt += 1

f1.close()



#view1:cite
eleft = list()
eright = list()
raw = set()
#get first view from cites. Ignore nodes that are not in node2id

f2 = open(filename)
num_edges_v1 = 0
for line in f2:
    values = line.split('\t')
    l_str = values[0].strip()
    r_str = values[1].strip()
    if(l_str not in node2id or r_str not in node2id or l_str == r_str):
        continue
    num_edges_v1 += 1
    l_int = node2id[l_str]
    r_int = node2id[r_str]
    raw.add((l_int,r_int))
    raw.add((r_int,l_int))

    #print(values)
f2.close()

#remove duplicates
for lr in raw:
    eleft.append(lr[0])
    eright.append(lr[1])


#get features from content

id_features = {}
f3 = open(filename_content)
for line in f3:
    values = line.split('\t')
    nd_str = values[0].strip()
    nd_id = node2id[nd_str]
    features_str = values[1:len(values)-1] #abandon last column, the class label
    features = list(map(float, features_str))
    id_features[nd_id] = features

f3.close()



#with input features
x = torch.zeros(len(id_features),len(id_features[0]))
for i in range(len(node2id)):
    x[i] = torch.FloatTensor(id_features[i])






eleft_v2 = list()
eright_v2 = list()
raw_v2 = set()

#get second view from view2_0.5 file, which is the output of generate_view.py
num_edges_v2 = 0
f4 = open(filename_v2)
for line in f4:
    values = line.split('\t')
    l_int = int(values[0].strip())
    r_int = int(values[1].strip())
    if(l_int == r_int):
        continue
    num_edges_v2 += 1
    raw_v2.add((l_int,r_int))
    raw_v2.add((r_int,l_int))


f4.close()

for lr in raw_v2:
    eleft_v2.append(lr[0])
    eright_v2.append(lr[1])




#run on view1 or view2

edge_index = torch.tensor([eleft,eright],dtype=torch.long)
edge_index_v2 = torch.tensor([eleft_v2,eright_v2],dtype=torch.long)


data = Data(x = x, edge_index = edge_index)
data_v2 = Data(x = x, edge_index = edge_index_v2)



class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        hidden_units = 128
        self.conv1 = GCNConv(in_channels, hidden_units, cached=True)#gcn
        self.conv2 = GCNConv(hidden_units, out_channels, cached=True)#gcn
  
    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))#gcn etc.
        return self.conv2(x, edge_index) #gcn etc.




num_features = len(data.x[0])





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MRGAE(Encoder(num_features, channels), Discriminator(channels)).to(device)


data.train_mask = data.val_mask = data.test_mask = data.y = None
data_v2.train_mask = data_v2.val_mask = data_v2.test_mask = data_v2.y = None


data = model.split_edges(data)
data_v2 = model.split_edges(data_v2,val_ratio=0.00, test_ratio=0.00)


x, train_pos_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)


train_pos_edge_index_v2 = data_v2.train_pos_edge_index.to(device)



model2 = copy.deepcopy(model)



#optimizer = torch.optim.Adam([{'params':model.parameters()},{'params':model2.parameters()}], lr = 0.001)  #pubmed
optimizer = torch.optim.Adam([{'params':model.parameters()},{'params':model2.parameters()}], lr = 0.0005) #other

def train():
    model.train()
    model2.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    z_v2 = model2.encode(x, train_pos_edge_index_v2)
    loss = (model.recon_loss_2views(z, train_pos_edge_index, z_v2, train_pos_edge_index_v2))
    loss = 1 * loss + 1* model.reg_loss(z) + 1 * (model.discriminator_loss(z) + 1 * model.discriminator_loss_2views(z,z_v2)) 
    loss.backward()
    optimizer.step()



def test(pos_edge_index, neg_edge_index):
    model.eval()
    model2.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)




best_auc = 0;
best_ap = 0;
best_epoch = 0;
for epoch in range(1, 300):#201
    #print('trainnnnnnnnnn')
    train()
    tauc, tap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
    if(tauc >= best_auc and tap >= best_ap):
        best_auc = tauc
        best_ap = tap
        best_epoch = epoch
    #auc, ap = test(data_v2.val_pos_edge_index, data_v2.val_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap), '; Best_Epoch: {:03d}, Best_AUC: {:.4f}, Best_AP: {:.4f}'.format(best_epoch, best_auc, best_ap))


auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)

print('Test AUC: {:.4f}, Test AP: {:.4f}'.format(auc, ap))



