import numpy as np 
import random

import torch 
from torch_geometric.utils.negative_sampling import structured_negative_sampling

 
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


def r2r_mat(r, num_users, num_movies):
    r_mat = torch.zeros(num_users, num_movies)
    r_mat[r[0], r[1]] = 1
    return r_mat
 
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


def bpr_loss(user_emb, user_emb_0, item_emb, item_emb_0, edge_index_r, batch_size = 128, lambda_= 1e-6):

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


def NDCG_K(model, train_r, test_r, K=20):
    '''
    https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1
    '''
    # Calculate the ratings for each user-item pair
    userEmbeddings = model.userEmb.weight
    itemEmbeddings = model.itemEmb.weight 
    ratings = userEmbeddings @ itemEmbeddings.T 

    # Set the ratings that are inside the training set to very negative number to ignore them
    ratings[train_r[0], train_r[1]] = -1e12 

    # For each user get the item ids(indices) that user positively interacted
    interaction_mat_test = r2r_mat(test_r, model.num_users, model.num_items) # shape: (num_users, num_items)
    pos_items_each_user_test = [row.nonzero().squeeze(1) for row in interaction_mat_test] 

    # Get top K recommended items (not their ratings but item ids) by the model, ratings are sorted in descending order
    _, topk_items_idxs_pred = torch.topk(ratings, k=K)

    # Turn those recommendation ids into binary, by preserving their recommended positions.
    rec_pred_binary = torch.zeros_like(topk_items_idxs_pred)

    for i in range(topk_items_idxs_pred.shape[0]):
        for j in range(topk_items_idxs_pred.shape[1]):
            if topk_items_idxs_pred[i,j] in pos_items_each_user_test[i]:
                # if the recommended item is in the list that user is positively interacted
                # meaning the recommendation is good
                rec_pred_binary[i,j] = 1

    # Turn positive items for each user into binary 2D array. 
    rec_gt_binary = torch.zeros_like(rec_pred_binary)
    for i in range(rec_gt_binary.shape[0]):
        l = min(len(pos_items_each_user_test[i]), K)
        rec_gt_binary[i, :l] = 1

    # Now calculate the NDGC
    idcg = (rec_gt_binary / torch.log2(torch.arange(K).float() + 2)).sum(dim=1)
    dcg = (rec_pred_binary / torch.log2(torch.arange(K).float() + 2)).sum(dim=1)

    ndcgs = dcg / idcg
    ndcg = ndcgs[~torch.isnan(ndcgs)]

    return ndcg.mean().item()
  
  