import streamlit as st
from rag.faiss_store import load_index_and_chunks, search
from rag.embedder import embed_text
from rag.llm import generate_answer

# Load index and chunks once at startup
@st.cache_resource
def load_data():
    return load_index_and_chunks()

index, chunks = load_data()

# App UI
st.set_page_config(page_title="📚 PDF Q&A Chat", layout="centered")
st.title("📚 Ask Questions from PDF")
st.markdown("This app lets you ask questions from the uploaded PDF using a local LLM via FAISS + embeddings.")

# Input box
question = st.text_input("🔎 Enter your question:")

if question:
    with st.spinner("🔍 Searching..."):
        vector = embed_text(question)
        if not vector:
            st.error("❌ Failed to generate embedding. Try again.")
        else:
            indices = search(index, vector)
            context = "\n\n".join([chunks[i] for i in indices])
            answer = generate_answer(context, question)

            # Display result
            st.markdown("### 🧠 Answer:")
            st.success(answer)

            # Optional: Show context
            with st.expander("🧾 Show retrieved context"):
                st.code(context)
