import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import os
import pickle as pkl
# ENGINE='torchsparse'
# ENGINE='dgl'
ENGINE=None
DGL_DEFER_CONVERSION=False

if ENGINE == 'dgl':
    import dgl.sparse as dglsp
elif ENGINE == 'torchsparse':
    from isplib import *
    # import torch_geometric.typing
    # torch_geometric.typing.WITH_PT2 = False
    # torch_geometric.typing.WITH_PT20 = False

def torch_sparse_to_dgl(torch_sparse):
    assert torch_sparse.is_sparse, "Input must be a sparse COO tensor"
    
    # Extract indices and values
    torch_sparse = torch_sparse.coalesce()
    
    indices = torch_sparse.indices()
    values = torch_sparse.values()
    shape = torch_sparse.shape
    
    # Convert to DGL SparseMatrix
    dgl_sparse = dglsp.spmatrix(indices, values, shape)
    return dgl_sparse


def perform_spmm(adj, dense, engine=ENGINE):
    if engine is None:
        return torch.sparse.mm(adj, dense)
    elif engine == 'torchsparse':
        return torch_sparse.matmul(adj, dense)
    elif engine == 'dgl':
        if DGL_DEFER_CONVERSION:
            with torch.no_grad():
                adj = torch_sparse_to_dgl(adj)
        return dglsp.spmm(adj, dense)

def save_mappings(filename, ent_map, rel_map):
  with open(filename, 'wb') as f:
    pkl.dump((ent_map, rel_map), f)

def load_mappings(filename):
  with open(filename, 'rb') as f:
    ent_map, rel_map = pkl.load(f)
  return ent_map, rel_map


def memory_mapped_tensor(filename, shape, dtype=torch.float32):
  # WARNING! It only works on CPU devices. For GPU, the memory mapping to disk will be ignored and a copy will be created.
  elem_size = torch.finfo(dtype).bits // 8
  total_bytes = np.prod(shape).item() * elem_size

  return torch.FloatTensor(
        torch.UntypedStorage.from_file(filename, shared=True, nbytes=total_bytes)
    ).reshape(shape)


def read_csv(filename, cols=['from', 'rel', 'to']):
   return pd.read_csv(filename, sep='\t',  names=cols, header=None)

def map_entity_and_rel(df, ent_map, rel_map):
  df['from'] = df['from'].map(ent_map)
  df['to'] = df['to'].map(ent_map)
  df['rel'] = df['rel'].map(rel_map)
  # return df

def generate_entity_map(df):
  # tmp = {j: i for i, j in enumerate(set(df['from'].unique()).union(set(df['to'].unique())))}
  alternating_entities = [None]*(len(df)*2)
  alternating_entities[::2] = df['from']
  alternating_entities[1::2] = df['to']

  alternating_entities = pd.Series(alternating_entities).unique()
  # assert len(tmp) == len(alternating_entities)
  # print(type(alternating_entities), alternating_entities.shape, len(tmp))
  return {j: i for i, j in enumerate(alternating_entities)}

def generate_rel_map(df):
  # return {j: i for i, j in enumerate(set(df['rel'].unique()))}
  return {j: i for i, j in enumerate(df['rel'].unique())}

def get_n_ent(df):
  # tmp = max(set(df['from']).union(set(df['to']))) + 1
  ret = len(set(df['from']).union(set(df['to'])))
  # assert tmp == ret
  return ret

def get_n_rel(df):
  # tmp = max(df['rel']) + 1
  ret = len(df['rel'].unique())
  # assert tmp == ret
  return ret

# def normalize_df(df, entity_map, rel_map):
#   df['from'] = df['from'].apply(lambda x: entity_map[x])
#   df['to'] = df['to'].apply(lambda x: entity_map[x])
#   df['rel'] = df['rel'].apply(lambda x: rel_map[x])
#   return df


def calculate_relation_statistics(df):
  def custom_agg(group):
      return pd.Series({
          'total_head_count': group['from'].count(),
          'unique_head_count': group['from'].nunique(),
          'total_tail_count': group['to'].count(),
          'unique_tail_count': group['to'].nunique()
      })
  df2 = df.groupby('rel').apply(custom_agg)
  df2['avg_heads_per_tail'] = df2['unique_head_count'] / df2['total_tail_count']
  df2['avg_tails_per_head'] = df2['unique_tail_count'] / df2['total_head_count']
  df2['total_avg'] = df2['avg_heads_per_tail'] + df2['avg_tails_per_head']
  df2['p_head'] = df2['avg_tails_per_head'] / df2['total_avg']
  df2['p_tail'] = df2['avg_heads_per_tail'] / df2['total_avg']

  # df['p_head'] = df['rel'].map(lambda x: df2.loc[x]['p_head'])
  # return df2[['p_head', 'p_tail']]
  return torch.tensor(df2['p_head'], dtype=torch.float32)


# def sparsify(n_ent, n_rel, df, h_idx=None, t_idx=None, r_idx=None, dtype=torch.float32):
#   if df is not None:
#     h_idx, t_idx, r_idx = torch.tensor(df['from']), torch.tensor(df['to']), torch.tensor(df['rel'])
#   bt_size = h_idx.shape[0]
#   device = h_idx.device
#   # n_ent = self.n_ent
#   # n_rel = self.n_rel
#   # print(f'{n_ent=}, {n_rel=}')
#   r_idx = r_idx.clone() + n_ent
#   adj_mat_idx = torch.tensor(range(bt_size), device=device).repeat(3)
#   adj_t = torch.sparse_coo_tensor(
#       indices=torch.stack([
#             adj_mat_idx,
#             torch.cat([h_idx, t_idx, r_idx])
#         ]),
#       values=torch.cat([
#             torch.full((len(h_idx),), 1, dtype=dtype, device=device),
#             torch.full((len(t_idx),), -1, dtype=dtype, device=device),
#             torch.full((len(r_idx),), 1, dtype=dtype, device=device),
#         ]),
#       size=(bt_size, n_ent + n_rel)
#   )
#   # return adj_t, df['p_head'].to_numpy(dtype=np.float16)
#   return adj_t


def generate_sparse_matrix(indices, values, size, engine=ENGINE):
    if engine is None:
        return torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=size
                )
    elif engine == 'torchsparse':
        # print(indices[1][32768*2:])
        return SparseTensor(
                row=indices[0],
                col=indices[1],
                value=values,
                sparse_sizes=size,
                # is_sorted=True
            ).requires_grad_(False)
    elif engine == 'dgl':
        if DGL_DEFER_CONVERSION:
            return torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=size
                )
        return dglsp.spmatrix(indices, values.requires_grad_(False), size)



def sparsify(n_ent, n_rel, df, dtype=torch.float32):
  concatenated = np.concatenate([df['from'].values, df['to'].values, df['rel'].values])

# Create tensor without data copy
  all_idx = torch.as_tensor(concatenated)
  # h_idx, t_idx, r_idx = torch.tensor(df['from']), torch.tensor(df['to']), torch.tensor(df['rel'])
  bt_size = df.shape[0]
  # device = h_idx.device
  # r_idx = r_idx.clone() + n_ent
  all_idx[2*bt_size:] += n_ent
    
  # adj_mat_idx = torch.tensor(range(bt_size), device=device).repeat(3)
  adj_mat_idx = torch.tile(torch.arange(bt_size), (3,))
  
  adj_t = generate_sparse_matrix(
      indices=torch.stack([
            adj_mat_idx,
            all_idx
        ]),
      values=torch.cat([
            torch.full((bt_size,), 1, dtype=dtype),
            torch.full((bt_size,), -1, dtype=dtype),
            torch.full((bt_size,), 1, dtype=dtype),
        ]),
      size=(bt_size, n_ent + n_rel)
      )
  # adj_t = dglsp.spmatrix(
  #     torch.stack([
  #           adj_mat_idx,
  #           all_idx
  #       ]),
  #     torch.cat([
  #           torch.full((bt_size,), 1, dtype=dtype),
  #           torch.full((bt_size,), -1, dtype=dtype),
  #           torch.full((bt_size,), 1, dtype=dtype),
  #       ]).requires_grad_(False),
  #     (bt_size, n_ent + n_rel)
  # )

  # adj_t = torch.sparse_coo_tensor(
  #     indices=torch.stack([
  #           adj_mat_idx,
  #           all_idx
  #       ]),
  #     values=torch.cat([
  #           torch.full((bt_size,), 1, dtype=dtype),
  #           torch.full((bt_size,), -1, dtype=dtype),
  #           torch.full((bt_size,), 1, dtype=dtype),
  #       ]),
  #     size=(bt_size, n_ent + n_rel)
  # )
    
    
  # return adj_t, df['p_head'].to_numpy(dtype=np.float16)
  return adj_t

class SparseKGDataset(Dataset):
    def __init__(self, df, batch_size, shuffle=False, drop_last=False, params=None, location=None, normalize=False, entity_map=None, rel_map=None, relation_stat=None, n_ent=None, n_rel=None):
        self.params = params
        self.location = location
        self.entity_map = entity_map or generate_entity_map(df)
        self.rel_map = rel_map or generate_rel_map(df)
        if normalize:
          map_entity_and_rel(df, self.entity_map, self.rel_map)
        self.n_ent = n_ent or get_n_ent(df)
        self.n_rel = n_rel or get_n_rel(df)
        self.relation_statistics = relation_stat or calculate_relation_statistics(df)
        # self.calculate_relation_statistics(df)
        # print(df['p_head'])
        if shuffle:
          df = df.sample(frac=1.0)
        if drop_last:
          self.dataset = [sparsify(self.n_ent, self.n_rel, df.iloc[i:i+batch_size].reset_index()) for i in range(0, len(df), batch_size)][:-1]
        else:
          self.dataset = [sparsify(self.n_ent, self.n_rel, df.iloc[i:i+batch_size].reset_index()) for i in range(0, len(df), batch_size)]
        self.dataset_size = len(self.dataset)
        # df = df.drop(['p_head'], axis=1)  

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.dataset[idx]


import pandas as pd
import torch
from torch import bernoulli

# def corrupt_batch_sparse(adj, n_ent, p_head=None, manual_seed=None):
#     if manual_seed is not None:
#       torch.manual_seed(manual_seed)
#       np.random.seed(manual_seed)
#     device = adj.device

#     expanded_size = adj.size()[0]
    
#     row_indice, col_indice = adj._indices()[0], adj._indices()[1].clone()
#     rel_items = col_indice[len(row_indice) // 3 * 2:] - n_ent
    
    
#     if p_head is not None:
#       head_mask = bernoulli(p_head[rel_items]).to(torch.bool)
#     else:
#       head_mask = bernoulli(torch.zeros_like(rel_items) + 0.5).to(torch.bool)
#     # print(head_mask)
    
#     tail_mask = ~head_mask
#     rel_mask = torch.zeros(expanded_size,).to(torch.bool)
#     all_mask = torch.cat([head_mask, tail_mask, rel_mask])

#     n_head_mask = head_mask.sum().item()
#     n_tail_mask = tail_mask.sum().item()
    
#     # rand_values = torch.cat([
#     #     torch.randint(1, n_ent, (n_head_mask,), device=device),
#     #     torch.randint(1, n_ent, (n_tail_mask,), device=device)
#     # ])
#     # rand_values = torch.randint(1, n_ent, (n_head_mask * 2,), device=device)

#     col_indice[all_mask] = torch.randint(1, n_ent, (n_head_mask + n_tail_mask,), device=device)

#     values = adj._values()
#     size = adj.size()[:]
    
#     return torch.sparse_coo_tensor(
#         indices=torch.stack([
#             row_indice,
#             col_indice
#         ]),
#         values=values,
#         size=size
#         )

def corrupt_batch_sparse(adj, n_ent, p_head=None, manual_seed=None):
    if manual_seed is not None:
      torch.manual_seed(manual_seed)
      np.random.seed(manual_seed)
    # device = adj.device
    
    
    if ENGINE == 'torchsparse':
      # print(adj.coo()[2][:10])
      # exit()
      reorder_tensor = lambda x: torch.cat([x[::3],x[1::3],x[2::3]])
      expanded_size = adj.sizes()[0]
      row_indice, col_indice, values = adj.coo()
      row_indice, col_indice, values = reorder_tensor(row_indice), reorder_tensor(col_indice), reorder_tensor(values)
      rel_items = col_indice[len(row_indice) // 3 * 2:] - n_ent
      size = adj.sizes()
    elif ENGINE is None or DGL_DEFER_CONVERSION: 
      expanded_size = adj.shape[0]
      row_indice, col_indice = adj._indices()[0], adj._indices()[1].clone()
      rel_items = col_indice[len(row_indice) // 3 * 2:] - n_ent
      values = adj._values()
      size = adj.size()[:]
    elif ENGINE == 'dgl': 
      expanded_size = adj.shape[0]
      row_indice, col_indice = adj.coo()[0], adj.coo()[1].clone()
      rel_items = col_indice[len(row_indice) // 3 * 2:] - n_ent
      values = adj.val
      size = adj.shape[:]
    
    if p_head is not None:
      head_mask = bernoulli(p_head[rel_items]).bool()
    else:
      head_mask = bernoulli(torch.full_like(rel_items, 0.5, dtype=torch.float32)).bool()
    
    tail_mask = ~head_mask
    rel_mask = torch.zeros(expanded_size, dtype=torch.bool)
    
    
    all_mask = torch.cat([head_mask, tail_mask, rel_mask])
    col_indice[all_mask] = torch.randint(1, n_ent, (expanded_size,))
    

    return generate_sparse_matrix(
      indices=torch.stack([
            row_indice,
            col_indice
        ]),
      values=values,
      size=size
      )


# Note: Adjusted for Torchsparse but not working.
# def corrupt_batch_sparse(adj, n_ent, p_head=None, manual_seed=None):
#     if manual_seed is not None:
#       torch.manual_seed(manual_seed)
#       np.random.seed(manual_seed)
#     # device = adj.device
    
#     if ENGINE is None:
#       expanded_size = adj.shape[0]
#       row_indice, col_indice = adj._indices()[0], adj._indices()[1].clone()
#       rel_items = col_indice[len(row_indice) // 3 * 2:] - n_ent
#       values = adj._values()
#       size = adj.size()[:]
#     elif ENGINE == 'torchsparse':
#       print(adj.coo()[2][:10])
#       exit()
#       expanded_size = adj.sizes()[0]
#       row_indice, col_indice = adj.coo()[0], adj.coo()[1].clone()
#       rel_items = col_indice[col_indice >= n_ent] - n_ent
#       values = adj.coo()[2]
#       size = adj.sizes()
#       rel_loc = (col_indice >= n_ent)
#       tail_loc = (values == -1)
#       # head_loc = ~(torch.zeros(len(rel_loc), dtype=torch.bool) & rel_loc & tail_loc)
#       head_loc = ~(rel_loc | tail_loc)
#       # print(len(row_indice))
#       # print(col_indice)
      
    
#     if p_head is not None:
#       head_mask = bernoulli(p_head[rel_items]).bool()
#     else:
#       head_mask = bernoulli(torch.full_like(rel_items, 0.5, dtype=torch.float32)).bool()
#       # head_mask = bernoulli(torch.zeros_like(rel_items) + 0.5).to(torch.bool)
#     # print(head_mask)
    
#     tail_mask = ~head_mask
#     rel_mask = torch.zeros(expanded_size, dtype=torch.bool)
    
#     if ENGINE != 'torchsparse':
#         all_mask = torch.cat([head_mask, tail_mask, rel_mask])
#         col_indice[all_mask] = torch.randint(1, n_ent, (expanded_size,))
#     else:
#         # all_mask = torch.cat([head_mask, tail_mask, rel_mask])
#         n_head_mask = head_mask.sum().item()
#         n_tail_mask = tail_mask.sum().item()
#         # print(head_loc)
#         # print(len(head_loc), len(tail_loc), len(rel_loc))
#         # print((rel_loc & tail_loc).sum().item(), head_loc.sum().item(), tail_loc.sum().item(), rel_loc.sum().item())
#         col_indice[head_loc][head_mask] = torch.randint(1, n_ent, (n_head_mask,))
#         col_indice[tail_loc][tail_mask] = torch.randint(1, n_ent, (n_tail_mask,))

#     return generate_sparse_matrix(
#       indices=torch.stack([
#             row_indice,
#             col_indice
#         ]),
#       values=values,
#       size=size
#       )
    


def corrupt_batch_triplets(triplets, n_ent, p_head=None, manual_seed=None):
    if manual_seed is not None:
      torch.manual_seed(manual_seed)
      np.random.seed(manual_seed)
    device = triplets.device

    # heads = triplets[:, 0]
    relations = triplets[:, 1]
    # tails = triplets[:, 2]
    if p_head is not None:
      head_mask = bernoulli(p_head[relations]).to(torch.bool)
    else:
      head_mask = bernoulli(torch.full_like(relations, 0.5, dtype=torch.float32)).bool()
      # head_mask = bernoulli(torch.zeros_like(relations) + 0.5).to(torch.bool)
    tail_mask = ~head_mask
    triplets_corrupt = triplets.clone()
    # print(head_mask)
    n_head_mask = head_mask.sum().item()
    n_tail_mask = tail_mask.sum().item()
    
    triplets_corrupt[head_mask, 0] = torch.randint(1, n_ent, (n_head_mask,), device=device)
    triplets_corrupt[tail_mask, 2] = torch.randint(1, n_ent, (n_tail_mask,), device=device)
    
    return triplets_corrupt[:, 0], triplets_corrupt[:, 1], triplets_corrupt[:, 2]

def repeat_adj(adj, n_neg):
  batch_size = adj.size()[0]
  expanded_size = batch_size * n_neg

  row_indice, col_indice = adj._indices()[0].clone(), adj._indices()[1].clone()
  new_values = adj._values().repeat_interleave(n_neg)
  new_size = (batch_size * n_neg, adj.size()[1])

  new_row_indice = torch.arange(expanded_size, device=row_indice.device).repeat(3)
  new_col_indice = col_indice.view(-1, batch_size).repeat(1, n_neg).view(-1)

  return torch.sparse_coo_tensor(
        indices=torch.stack([
            new_row_indice,
            new_col_indice
        ]),
        values=new_values,
        size=new_size
        )
