import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch 

from tqdm import tqdm
import matplotlib.pyplot as plt

from model import LightGCN
from utils import *



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


train_edges = int_edges[:, train_idx]
test_edges  = int_edges[:, test_idx]
 
train_adj = create_adj_matrix(train_edges, num_users, num_movies)
test_adj  = create_adj_matrix(test_edges, num_users, num_movies)

train_r = adj_to_r_mat(train_adj, num_users, num_movies)
test_r  = adj_to_r_mat(test_adj, num_users, num_movies)

  
''' ------------ Training Loop ------------ '''

NUM_ITER   = 10000
BATCH_SIZE = 512

# model = LightGCN(num_users, num_movies)
# model.load_state_dict(torch.load('trained_model.pth'))

model = LightGCN(num_users, num_movies)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

loss_lst = []
val_loss_lst = []
ndcg_k_lst = []

iterator = tqdm(range(NUM_ITER))
val_loss = 0
ndcg_k = 0
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

    if (i+1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            user_emb, user_emb_0, item_emb, item_emb_0 = model(test_adj)
        loss_val = bpr_loss(
            user_emb, 
            user_emb_0, 
            item_emb, 
            item_emb_0, 
            test_r,
            BATCH_SIZE
            )
        val_loss = loss_val.item()
        val_loss_lst.append(val_loss)

        ndcg_k = NDCG_K(model, train_r, test_r, K=20)
        ndcg_k_lst.append(ndcg_k)


    iterator.set_postfix({
        'Train Loss': loss.item(),
        'Val Loss': val_loss,
        'NDCG_K': ndcg_k
        })
    loss_lst.append(loss.item())
    
    if i % 200 == 0 and i != 0:
        scheduler.step()


torch.save(model.state_dict(), 'trained_model2.pth')

plt.style.use("ggplot")


x_axis = [i * 100 for i in range(len(val_loss_lst))]
plt.plot(x_axis, np.array(loss_lst)[x_axis], label='Training Loss')
plt.plot(x_axis, val_loss_lst, label='Validation Loss')
plt.title("Training and Validation Loss Curves")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
fig = plt.gcf()
fig.savefig("loss_plot.png", dpi=300)
plt.show()


  
x_axis = [i * 100 for i in range(len(val_loss_lst))]
plt.plot(x_axis, ndcg_k_lst, label='Validation ndcg')
plt.title(f"Normalized Discounted Cumulative Gain, K={20}")
plt.xlabel("Iteration")
plt.ylabel("ndcg metric")
plt.legend()
fig = plt.gcf()
fig.savefig("ndcg.png", dpi=300)
plt.show()

  
  