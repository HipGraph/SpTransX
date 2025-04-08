import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_
from fastkg.utils import memory_mapped_tensor, perform_spmm

class SparseTransE(nn.Module):
    def __init__(self, n_ent, n_rel, emb_size, storage=None, norm_order=2, initialize=True):
        super(SparseTransE, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.norm_order = norm_order
        self.emb_size = emb_size
        self.memory_mapped = False
        
        if storage is not None:
            # float_byte_size = torch.finfo(torch.float32).bits // 8
            # self.all_emb = nn.Parameter(torch.FloatTensor(
            #     torch.UntypedStorage.from_file(storage, shared=True, nbytes=((n_ent + n_rel) * emb_size * float_byte_size))
            # ).reshape(n_ent + n_rel, emb_size))
            # self.tmp = memory_mapped_tensor(storage, (n_ent + n_rel, emb_size))
            # self.all_emb = nn.Parameter(self.tmp)
            # self.all_emb = self.tmp.requires_grad_()
            self.all_emb = nn.Parameter(memory_mapped_tensor(storage, (n_ent + n_rel, emb_size)))
            self.memory_mapped = True
        else:
            self.all_emb = nn.Parameter(torch.empty(n_ent + n_rel, emb_size))
        
        if initialize:
            self.initialize_entity_embeddings()
            self.initialize_rel_embeddings()
          
    def forward(self, adj_t, adj_t2, t_idx=None, valid=False):
        self.normalize_entity_weights()
        if not valid:
            pos_calc = -1 * torch.linalg.vector_norm(perform_spmm(adj_t, self.all_emb), ord=self.norm_order, dim=1)**2
            neg_calc = -1 * torch.linalg.vector_norm(perform_spmm(adj_t2, self.all_emb), ord=self.norm_order, dim=1)**2
            return pos_calc, neg_calc
        else:
            return self.calculate_score(adj_t, adj_t2, t_idx) # adj_t <- h_idx, adj_t2 <- r_idx
    
    def initialize_entity_embeddings(self):
        with torch.no_grad():
            if self.memory_mapped:
                self.all_emb[:self.n_ent, :].copy_(xavier_uniform_(torch.randn(self.n_ent, self.emb_size)))
            else:
                self.all_emb.data[:self.n_ent, :] = xavier_uniform_(torch.randn(self.n_ent, self.emb_size))
        self.normalize_entity_weights()
    
    def initialize_rel_embeddings(self):
        with torch.no_grad():
            if self.memory_mapped:
                self.all_emb[self.n_ent:, :].copy_(xavier_uniform_(torch.randn(self.n_rel, self.emb_size)))
            else:
                self.all_emb.data[self.n_ent:, :] = xavier_uniform_(torch.randn(self.n_rel, self.emb_size))
        self.normalize_rel_weights()
    
    def normalize_all_emb(self):
      self.normalize_entity_weights()
      self.normalize_rel_weights()

    def normalize_entity_weights(self):
        with torch.no_grad():
            if self.memory_mapped:
                self.all_emb[:self.n_ent, :].copy_(normalize(self.all_emb[:self.n_ent, :], p=self.norm_order, dim=1))
            else:
                self.all_emb.data[:self.n_ent, :] = normalize(self.all_emb.data[:self.n_ent, :], p=self.norm_order, dim=1)

    def normalize_rel_weights(self):
        with torch.no_grad():
            if self.memory_mapped:
                self.all_emb[self.n_ent:, :].copy_(normalize(self.all_emb[self.n_ent:, :], p=self.norm_order, dim=1))
            else:
                self.all_emb.data[self.n_ent:, :] = normalize(self.all_emb.data[self.n_ent:, :], p=self.norm_order, dim=1)

    def get_entity_weights(self):
        return self.all_emb[:self.n_ent, :]

    def get_rel_weights(self):
        return self.all_emb[self.n_ent:, :]
    
    def calculate_score(self, h_idx, r_idx, t_idx):
        head = self.all_emb[h_idx]
        tail = self.all_emb[t_idx]
        relation = self.all_emb[r_idx + self.n_ent]
        return torch.linalg.vector_norm(head + relation - tail, ord=self.norm_order, dim=-1)**2

    def classify_triplet(self, h_idx, r_idx, t_idx, threshold=1.0):
        with torch.no_grad():
            score = self.calculate_score(h_idx, r_idx, t_idx)
            return score.item() < threshold
        
    def predict_tail(self, h_idx, r_idx):
        scores = []
        for t_idx in range(self.n_ent):
            score = self.calculate_score(h_idx, r_idx, t_idx)
            scores.append(score.item())
        scores = torch.tensor(scores)
        top_k = scores.topk(k=1, largest=False)
        return top_k.indices.item()
    
    def predict_head(self, t_idx, r_idx):
        scores = []
        for h_idx in range(self.n_ent):
            score = self.calculate_score(h_idx, r_idx, t_idx)
            scores.append(score.item())
        scores = torch.tensor(scores)
        top_k = scores.topk(k=1, largest=False)
        return top_k.indices.item()

    