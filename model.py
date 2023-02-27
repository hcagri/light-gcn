import torch 
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.negative_sampling import structured_negative_sampling
from torch_geometric.transforms import gcn_norm


class LightGCNConv(MessagePassing):
    ''' How message passing base class works?
        https://zqfang.github.io/2021-08-07-graph-pyg/#messagepassing-in-pytorch-geometric 
        j : Source index
        i : Target index
    '''
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        norm = gcn_norm(edge_index)
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    

class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, emb_dim = 64, num_layers=4):
        super().__init__()
        
        self.userEmb = nn.Embedding(num_users, emb_dim)
        self.itemEmb = nn.Embedding(num_items, emb_dim)

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(LightGCNConv())

    def forward(self, edge_index):
        emb = torch.cat([self.userEmb, self.itemEmb], dim=0) 
        emb_lst = [emb]

        for lightgcn in self.convs:
            emb = lightgcn(emb, edge_index)
            emb_lst.append(emb_lst)
        
        out_emb = torch.stack(emb_lst, dim=1)   
        out_emb = torch.mean(out_emb, dim=1)

        users_emb_final, items_emb_final = torch.split(out_emb, [self.num_users, self.num_items]) 

        return users_emb_final, items_emb_final


