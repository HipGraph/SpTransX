import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import sqlite3, random
from fastkg.utils import sparsify
from math import ceil


class SQLiteConnection:
    def __init__(self, db_name):
        self.db_name = db_name

    def __enter__(self):
        # print('connecting...')
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        return self.conn

    def __exit__(self, exc_type, exc_value, traceback):
        # print('disconnecting...')
        self.conn.close()

# def get_db_conn(db_name):
#     conn = sqlite3.connect(db_name)
#     return conn

def get_total_entities_and_relations(db_name):
    with SQLiteConnection(db_name) as conn:
        cursor = conn.cursor()
        
        # Get the total number of entities
        cursor.execute('SELECT COUNT(*) FROM entities')
        total_entities = cursor.fetchone()[0]
        
        # Get the total number of relations
        cursor.execute('SELECT COUNT(*) FROM relations')
        total_relations = cursor.fetchone()[0]
        
        return total_entities, total_relations

def get_total_triplet_count(db_name):
    with SQLiteConnection(db_name) as conn:
        cursor = conn.cursor()
        # Get the total number of entities
        cursor.execute('SELECT COUNT(*) FROM triplets')
        total_triplets = cursor.fetchone()[0]
        
        return total_triplets

def get_entity_id(db_name, entity, relation=False):
    with SQLiteConnection(db_name) as conn:
        cursor = conn.cursor()
        if not relation:
            cursor.execute('SELECT entity_id FROM entities WHERE entity = ?', (entity,))
        else:
            cursor.execute('SELECT relation_id FROM relations WHERE relation = ?', (entity,))
        result = cursor.fetchone()
        if result:
            return result[0] - 1
        else:
            return None

def get_entity_str(db_name, entity_id, relation=False):
    with SQLiteConnection(db_name) as conn:
        cursor = conn.cursor()
        entity_id = entity_id + 1
        if not relation:
            cursor.execute('SELECT entity FROM entities WHERE entity_id = ?', (entity_id,))
        else:
            cursor.execute('SELECT relation FROM relations WHERE relation_id = ?', (entity_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return None

def get_entity_id_map(db_name, entities, relation=False):
    with SQLiteConnection(db_name) as conn:
        cursor = conn.cursor()
        # Prepare placeholders for the SQL query
        placeholders = ', '.join(['?'] * len(entities))
        
        # Choose the correct table and column based on the relation flag
        if not relation:
            query = f'SELECT entity_id, entity FROM entities WHERE entity IN ({placeholders})'
        else:
            query = f'SELECT relation_id, relation FROM relations WHERE relation IN ({placeholders})'
        
        # Execute the query with the array of entities
        cursor.execute(query, entities)
        
        # Fetch all results
        results = cursor.fetchall()
        # print(results)
        # Create a dictionary mapping entities/relations to their IDs
        ret = {row[1]: row[0] - 1 for row in results}
        
        # Return the dictionary of IDs
        return ret
    

def get_all_entities(db_name):
    with SQLiteConnection(db_name) as conn:
        cursor = conn.cursor()
        # cursor.execute('SELECT * FROM triplets WHERE id BETWEEN ? AND ?', (start, end))
        cursor.execute('SELECT entity FROM entities')
        ret = cursor.fetchall()
        # print(ret)
        # print('after shuffle:')
        # random.shuffle(ret)
        # print(ret)
        
        # return torch.tensor(ret)
        return ret

def get_triplets_range(db_name, start, end, shuffle=False):
    with SQLiteConnection(db_name) as conn:
        start = start + 1
        end = end
        cursor = conn.cursor()
        # cursor.execute('SELECT * FROM triplets WHERE id BETWEEN ? AND ?', (start, end))
        cursor.execute('SELECT subject_id, predicate_id, object_id FROM triplets WHERE id BETWEEN ? AND ?', (start, end))
        ret = cursor.fetchall()
        # print(ret)
        # print('after shuffle:')
        # random.shuffle(ret)
        # print(ret)
        if shuffle:
            random.shuffle(ret)
        # return torch.tensor(ret)
        return pd.DataFrame(ret, columns=['from', 'rel', 'to'])


# def read_csv(filename, cols=['from', 'rel', 'to']):
#    return pd.read_csv(filename, sep='\t',  names=cols, header=None)

def map_entity_and_rel(df, db_name):
    # with SQLiteConnection(db_name) as conn:
    val_entities = set(df['from'].unique()).union(set(df['to'].unique()))
    val_rels = set(df['rel'].unique())
    ent_map = get_entity_id_map(db_name, list(val_entities))
    rel_map = get_entity_id_map(db_name, list(val_rels), relation=True)
    
    df['from'] = df['from'].map(ent_map)
    df['to'] = df['to'].map(ent_map)
    df['rel'] = df['rel'].map(rel_map)
    # return df


# Function to calculate relation statistics and store in SQLite
def calculate_relation_statistics_sqlite(db_name):
    with SQLiteConnection(db_name) as conn:
        cursor = conn.cursor()

        # Define SQL query to calculate statistics
        query = """
            SELECT 
                predicate_id As rel,
                COUNT("subject_id") AS total_head_count,
                COUNT(DISTINCT "subject_id") AS unique_head_count,
                COUNT("object_id") AS total_tail_count,
                COUNT(DISTINCT "object_id") AS unique_tail_count
            FROM triplets
            GROUP BY predicate_id
        """

        # Execute query and fetch results
        cursor.execute(query)
        rows = cursor.fetchall()

        # Initialize lists to store calculated values
        p_head_values = []
        p_tail_values = []

        # Calculate derived statistics and insert into the database
        for row in rows:
            rel, total_head_count, unique_head_count, total_tail_count, unique_tail_count = row

            avg_heads_per_tail = unique_head_count / total_tail_count if total_tail_count > 0 else 0.0
            avg_tails_per_head = unique_tail_count / total_head_count if total_head_count > 0 else 0.0
            total_avg = avg_heads_per_tail + avg_tails_per_head
            p_head = avg_tails_per_head / total_avg if total_avg > 0 else 0.0
            p_tail = avg_heads_per_tail / total_avg if total_avg > 0 else 0.0

            p_head_values.append((rel, p_head))
            p_tail_values.append((rel, p_tail))
        # Return p_head tensor for further processing (if needed)
        p_head_tensor = torch.tensor([p_head for _, p_head in p_head_values], dtype=torch.float32)
        return p_head_tensor


class StreamingSparseKGDataset(Dataset):
    def __init__(self, db_name, batch_size, shuffle=False, drop_last=False, params=None, location=None, relation_stat=None, n_ent=None, n_rel=None, calculate_rel_stat=False):
        self.params = params
        self.location = location
        self.db_name = db_name
        
        if n_ent is None or n_rel is None:
            stat = get_total_entities_and_relations(db_name)
        else:
            stat = (n_ent, n_rel)
        self.n_ent, self.n_rel = stat
        self.total_facts = get_total_triplet_count(db_name)
        if calculate_rel_stat:
            self.relation_statistics = relation_stat or calculate_relation_statistics_sqlite(db_name)
        else:
            self.relation_statistics = None
        
        self.batch_size = batch_size
        self.perform_shuffle = shuffle
        if drop_last:
            self.dataset_size = ceil(self.total_facts / self.batch_size) - 1
        else:
            self.dataset_size = ceil(self.total_facts / self.batch_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx < 0:
            idx = idx + self.dataset_size
        if idx >= self.dataset_size or idx < 0:
            raise IndexError(f"Index {idx} out of range.")
        batch_data_db = get_triplets_range(self.db_name, idx * self.batch_size, (idx + 1) * self.batch_size)
        sparse_adj = sparsify(self.n_ent, self.n_rel, batch_data_db)
        return sparse_adj

