 
import pandas as pd 
import numpy as np 
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch_geometric as pyg
from torch_geometric.utils.negative_sampling import structured_negative_sampling
from torch_geometric.transforms import gcn_norm

from model import LightGCN

 
def create_interaction_edges(userids, movieids, ratings_, threshold=3.5):
    ''' Interaction edges in COO format.'''
    mask = ratings_ > threshold
    edges = np.stack([userids[mask], movieids[mask]])
    return torch.LongTensor(edges)

 
def create_adj_matrix(int_edges, num_users, num_movies):

    n = num_users + num_movies
    adj = torch.zeros(n,n)

    r_mat = torch.sparse_coo_tensor(int_edges, torch.ones(int_edges.shape[1]), size=(num_users, num_movies)).to_dense()
    adj[:num_users, num_users:] = r_mat.clone()
    adj[num_users:, :num_users] = r_mat.T.clone()

    adj_coo = adj.to_sparse_coo()
    adj_coo = adj_coo.indices()

    return adj_coo

 
def adj_to_r_mat(adj, num_users, num_movies):
    n = num_users + num_movies
    adj_dense = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), size=(n, n)).to_dense()
    r_mat = adj_dense[:num_users, num_users:]
    r_coo = r_mat.to_sparse_coo()
    return r_coo.indices()
 
## edges = (i, j, k) (i, j) positive edge (i, k) negative edge
def sample_mini_batch(edge_index, batch_size):

    '''
    Args:
        edge_index: edge_index of the user-item interaction matrix
    
    Return:
        structured_negative_sampling return (i,j,k) where
            (i,j) positive edge
            (i,k) negative edge
    '''
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    random_idx = random.choices(list(range(edges[0].shape[0])), k=batch_size)
    batch = edges[:, random_idx]
    user_ids, pos_items, neg_items = batch[0], batch[1], batch[2]
    return user_ids, pos_items, neg_items


def bpr_loss(user_emb, user_emb_0, item_emb, item_emb_0, edge_index_r, batch_size = 128, lambda_= 1e-4):

    user_ids, pos_items, neg_items = sample_mini_batch(edge_index_r, batch_size=batch_size)

    user_emb_sub = user_emb[user_ids]
    pos_item_emb = item_emb[pos_items]
    neg_item_emb = item_emb[neg_items]

    pos_scores = torch.diag(user_emb_sub @ pos_item_emb.T)
    neg_scores = torch.diag(user_emb_sub @ neg_item_emb.T)

    reg_loss = lambda_*(
        user_emb_0[user_ids].norm(2).pow(2) +
        item_emb_0[pos_items].norm(2).pow(2) +
        item_emb_0[neg_items].norm(2).pow(2) # L2 loss
    )

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss
    return loss
    

ratings = pd.read_csv('ml-latest-small/ratings.csv')
 
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['userId']  = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

num_users  = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()


int_edges = create_interaction_edges(ratings['userId'], ratings['movieId'], ratings['rating'])
 
indices = torch.arange(0, int_edges.shape[1], dtype=torch.long)

train_idx, test_idx = train_test_split(indices, test_size=0.2)
val_idx, test_idx   = train_test_split(test_idx, test_size=0.5)

train_edges = int_edges[:, train_idx]
val_edges   = int_edges[:, val_idx]
test_edges  = int_edges[:, test_idx]
 
train_adj = create_adj_matrix(train_edges, num_users, num_movies)
val_adj   = create_adj_matrix(val_edges, num_users, num_movies)
test_adj  = create_adj_matrix(test_edges, num_users, num_movies)


''' ------------ Training Loop ------------ '''

train_r = adj_to_r_mat(train_adj, num_users, num_movies)
val_r   = adj_to_r_mat(val_adj, num_users, num_movies)
test_r  = adj_to_r_mat(test_adj, num_users, num_movies)

NUM_ITER   = 1000
BATCH_SIZE = 512

model = LightGCN(num_users, num_movies)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


loss_lst = []
val_loss_lst = []

iterator = tqdm(range(NUM_ITER))
val_loss = 0
for i in iterator:

    model.train()
    optimizer.zero_grad()

    user_emb, user_emb_0, item_emb, item_emb_0 = model(train_adj)
    loss = bpr_loss(
        user_emb, 
        user_emb_0, 
        item_emb, 
        item_emb_0, 
        train_r,
        BATCH_SIZE
        )

    loss.backward()
    optimizer.step()

    if (i+1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            user_emb, user_emb_0, item_emb, item_emb_0 = model(val_adj)
        loss_val = bpr_loss(
            user_emb, 
            user_emb_0, 
            item_emb, 
            item_emb_0, 
            val_r,
            BATCH_SIZE
            )
        val_loss = loss_val.item()
        val_loss_lst.append(val_loss)

    iterator.set_postfix({
        'Train Loss': loss.item(),
        'Val Loss': val_loss
        })
    loss_lst.append(loss.item())


plt.style.use("ggplot")
plt.plot(loss_lst)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
fig = plt.gcf()
fig.savefig("loss_plot.png", dpi=300)
plt.show()

