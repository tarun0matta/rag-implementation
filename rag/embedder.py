import requests

def embed_text(text, model="nomic-embed-text:v1.5"):
    response = requests.post("http://localhost:11434/api/embeddings", json={"model": model, "prompt": text})
    data = response.json()
    return data.get("embedding")

def embed_chunks(chunks):
    return [embed_text(chunk) for chunk in chunks]
