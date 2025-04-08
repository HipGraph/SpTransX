# (SparseTransX) FastKG - A sparse implementation of translational KG embedding models

This is the official implementation of the SparseTransX library accepted for publication in MLSys 2025, the 8th Annual Conference on Machine Learning and Systems. "SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations". 

arXiv: https://arxiv.org/abs/2502.16949

## Installation
    
    git clone https://github.com/HipGraph/SpTransX.git
    cd SpTransX
    pip install -e .

# CPU/GPU Testing
To test fb15k dataset:

    cd ./tests
    python trans_e.py

# MultiGPU/MultiNode Testing 

FastKG is compatible with PyTorch DDP and FSDP Wrapper. They can be utilized to perform MultiGPU/MultiNode training.

# Streaming Dataset and Model

FastKG supports streaming both model and dataset from disk in case they are too large to fit in CPU memory. The streaming is also available for distributed training. Examples are available below.

## CPU/GPU

See the example in `./tests/trans_e_stream_dataset.py` and `./tests/trans_e_stream_model.py` on how to stream dataset and model on-demand instead of loading the whole in CPU memory.

    cd ./tests/
    python trans_e_stream_dataset.py
    # or, 
    python trans_e_stream_model.py

### Streaming Dataset

Create a `StreamingSparseKGDataset` instead of `SparseKGDataset`.

    convert_nt_to_db('../fastkg/datasets/fb15k/train.txt', batch_size=50000, num_lines=None, sep='\t', db_name='fb15k.db')
    dataset_sparse = StreamingSparseKGDataset('fb15k.db', batch_size=b_size, shuffle=False, drop_last=False, calculate_rel_stat=True)
    # For validation:
    from fastkg.streaming import map_entity_and_rel
    map_entity_and_rel(df_test, 'fb15k.db')

### Streaming Model

Pass a filename in storage argument when creating `SparseTransE` model.

    model_sparse = SparseTransE(dataset_sparse.n_ent, dataset_sparse.n_rel, emb_dim, storage='embedding_tensor.bin', initialize=True)
    model_sparse.to(device)



> [!NOTE]  
> Please note that some systems may not support memory mapped tensor (mmap is required for streaming the model since it uses memory mapped tensor) such as DVS in NERSC supercomputer. For NERSC, it is advised to use $PSCRATCH instead.
  
