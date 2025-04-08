import os
import pandas as pd
import numpy as np
from fastkg.utils import read_csv

__all__ = ['load_fb15k', 'load_covid19']

def load_fb15k(category='train'):
  current_dir = os.path.dirname(os.path.abspath(__file__))  
  file_path = os.path.join(current_dir, 'fb15k', f'{category}.txt')
  return read_csv(file_path)

def load_covid19(category='train'):
  current_dir = os.path.dirname(os.path.abspath(__file__))  
  file_path = os.path.join(current_dir, 'covid19', f'{category}.txt')
  return read_csv(file_path)

def generate_dummy_dataset(n_ent, n_rel, n_facts, manual_seed=0):
  np.random.seed(manual_seed)
  return pd.DataFrame({
      'from': np.random.randint(0, n_ent, (n_facts,)),
      'rel': np.random.randint(0, n_rel, (n_facts,)),
      'to': np.random.randint(0, n_ent, (n_facts,))
  })
# generate_dummy_dataset(10, 3, 100)
