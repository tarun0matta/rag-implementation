import faiss
import numpy as np
import json

def store_embeddings(embeddings, path="index/faiss_index.index"):
    matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, path)

def save_chunks(chunks, path="index/chunks.json"):
    with open(path, "w") as f:
        json.dump(chunks, f)

def load_index_and_chunks(index_path="index/faiss_index.index", chunks_path="index/chunks.json"):
    index = faiss.read_index(index_path)
    with open(chunks_path) as f:
        chunks = json.load(f)
    return index, chunks

def search(index, vector, k=3):
    query_np = np.array([vector]).astype("float32")
    _, indices = index.search(query_np, k)
    return indices[0]
