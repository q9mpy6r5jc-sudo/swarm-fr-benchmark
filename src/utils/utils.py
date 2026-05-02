import scipy.sparse as sp
import pandas as pd
import os, glob, re
from tqdm import tqdm

def load_sparse_from_csv(filepath, chunk_size=10000):
    """Reads a massive CSV in chunks and compresses it to a sparse matrix."""
    sparse_blocks = []
    for chunk in pd.read_csv(filepath, header=None, chunksize=chunk_size):
        sparse_blocks.append(sp.csr_matrix(chunk.values))

    return sp.vstack(sparse_blocks, format='csr')