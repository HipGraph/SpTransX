import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TripletDataset(Dataset):
    def __init__(self, dataframe, device):
        self.data = dataframe.values
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.data[idx], dtype=torch.long, device=self.device)


def calculate_hits_at_k_batch(model, df, batch_size, device, k=10, norm_order=2):
    dataset = TripletDataset(df, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    head_hits = 0
    tail_hits = 0
    num_triples = len(dataset)

    # Extract entity and relation embeddings
    entity_embeddings = model.get_entity_weights()
    relation_embeddings = model.get_rel_weights()
    
    # print(entity_embeddings[1,:])

    for batch_triplets in tqdm(dataloader):
        heads = batch_triplets[:, 0]  # Extract heads from triplets
        relations = batch_triplets[:, 1]  # Extract relations from triplets
        tails = batch_triplets[:, 2]  # Extract tails from triplets

        # Get the embeddings for all heads, tails, and relations in the batch
        head_embeddings = entity_embeddings[heads]
        tail_embeddings = entity_embeddings[tails]
        relation_embeddings_batch = relation_embeddings[relations]

        # Calculate the scores for all possible tails in batch (corrupting tails)
        all_tail_scores = torch.norm(head_embeddings.unsqueeze(1) + relation_embeddings_batch.unsqueeze(1) - entity_embeddings.unsqueeze(0), p=norm_order, dim=2)

        # Calculate the scores for all possible heads in batch (corrupting heads)
        all_head_scores = torch.norm(entity_embeddings.unsqueeze(0) + relation_embeddings_batch.unsqueeze(1) - tail_embeddings.unsqueeze(1), p=norm_order, dim=2)

        # Get the top k scores for each triplet in the batch for tail corruption
        topk_tail_scores, topk_tail_indices = torch.topk(all_tail_scores, k, largest=False, dim=1)

        # Get the top k scores for each triplet in the batch for head corruption
        topk_head_scores, topk_head_indices = torch.topk(all_head_scores, k, largest=False, dim=1)

        # Check if the true tails are in the top k indices for each triplet (tail corruption)
        true_tail_mask = (topk_tail_indices == tails.unsqueeze(1))  # Mask indicating if true tails are in top k for each triplet
        tail_hits += true_tail_mask.sum().item()  # Total number of hits for tails

        # Check if the true heads are in the top k indices for each triplet (head corruption)
        true_head_mask = (topk_head_indices == heads.unsqueeze(1))  # Mask indicating if true heads are in top k for each triplet
        head_hits += true_head_mask.sum().item()  # Total number of hits for heads

    # Calculate Hits@k for heads and tails
    hits_at_k_heads = head_hits / num_triples
    hits_at_k_tails = tail_hits / num_triples

    return hits_at_k_heads, hits_at_k_tails

def calculate_hits_at_k(model, triplets, k=10, norm_order=2):
    head_hits = 0
    tail_hits = 0
    num_triples = len(triplets)

    # Extract entity and relation embeddings
    entity_embeddings = model.get_entity_weights()
    relation_embeddings = model.get_rel_weights()

    heads = triplets[:, 0]  # Extract heads from triplets
    relations = triplets[:, 1]  # Extract relations from triplets
    tails = triplets[:, 2]  # Extract tails from triplets

    # Get the embeddings for all heads, tails, and relations in the batch
    head_embeddings = entity_embeddings[heads]
    tail_embeddings = entity_embeddings[tails]
    relation_embeddings = relation_embeddings[relations]

    # Calculate the scores for all possible tails in batch (corrupting tails)
    all_tail_scores = torch.norm(head_embeddings.unsqueeze(1) + relation_embeddings.unsqueeze(1) - entity_embeddings.unsqueeze(0), p=norm_order, dim=2)

    # Calculate the scores for all possible heads in batch (corrupting heads)
    # all_head_scores = torch.norm(entity_embeddings.unsqueeze(0).unsqueeze(2) + relation_embeddings.unsqueeze(0) - tail_embeddings.unsqueeze(1), p=2, dim=2)
    all_head_scores = torch.norm(entity_embeddings.unsqueeze(0) + relation_embeddings.unsqueeze(1) - tail_embeddings.unsqueeze(1), p=norm_order, dim=2)

    # if filt:
    #     # Create a mask to filter out true triplets from the scores for tail corruption
    #     for i, (h, t, r) in enumerate(triplets):
    #         all_true_tails = set(torch.where((triplets[:, 0] == h) & (triplets[:, 2] == r))[0].tolist())
    #         all_true_tails.discard(i)  # Remove the current triplet
    #         all_tail_scores[i, list(all_true_tails)] = float('inf')

    #     # Create a mask to filter out true triplets from the scores for head corruption
    #     for i, (h, t, r) in enumerate(triplets):
    #         all_true_heads = set(torch.where((triplets[:, 1] == t) & (triplets[:, 2] == r))[0].tolist())
    #         all_true_heads.discard(i)  # Remove the current triplet
    #         all_head_scores[i, list(all_true_heads)] = float('inf')

    # Get the top k scores for each triplet in the batch for tail corruption
    topk_tail_scores, topk_tail_indices = torch.topk(all_tail_scores, k, largest=False, dim=1)

    # Get the top k scores for each triplet in the batch for head corruption
    topk_head_scores, topk_head_indices = torch.topk(all_head_scores, k, largest=False, dim=1)

    # Check if the true tails are in the top k indices for each triplet (tail corruption)
    true_tail_mask = (topk_tail_indices == tails.unsqueeze(1))  # Mask indicating if true tails are in top k for each triplet
    tail_hits = true_tail_mask.sum().item()  # Total number of hits for tails

    # Check if the true heads are in the top k indices for each triplet (head corruption)
    true_head_mask = (topk_head_indices == heads.unsqueeze(1))  # Mask indicating if true heads are in top k for each triplet
    head_hits = true_head_mask.sum().item()  # Total number of hits for heads

    # Calculate Hits@k for heads and tails
    hits_at_k_heads = head_hits / num_triples
    hits_at_k_tails = tail_hits / num_triples

    return hits_at_k_heads, hits_at_k_tails

