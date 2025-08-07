# src/vector_store.py
import os
import numpy as np
import faiss

FAISS_INDEX_FILE = "data/embeddings/faiss.index"
CHUNKS_FILE = "data/embeddings/chunks.npy"

def create_index(dim):
    return faiss.IndexFlatL2(dim)

def save_embeddings(vectors, chunks):
    vectors_np = np.array(vectors, dtype="float32")
    index = create_index(vectors_np.shape[1])
    index.add(vectors_np)
    
    os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks, dtype=object))

def load_embeddings():
    index = faiss.read_index(FAISS_INDEX_FILE)
    chunks = np.load(CHUNKS_FILE, allow_pickle=True)
    return index, chunks

def search(query_vector, index, chunks, top_k=3):
    query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
    D, I = index.search(query_vector, top_k)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
