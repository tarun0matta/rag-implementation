from rag.pdf_loader import pdf_to_text
from rag.chunker import chunk_text
from rag.embedder import embed_chunks
from rag.faiss_store import store_embeddings, save_chunks

if __name__ == "__main__":
    text = pdf_to_text("data/awsbook.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_embeddings(embeddings)
    save_chunks(chunks)
    print(" PDF processed and indexed.")
