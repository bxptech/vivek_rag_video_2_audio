# src/rag_query.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from src.vector_store import search

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def rag_query(query_text, embed_func, index, chunks):
    # Get embedding for query
    query_vec = embed_func(query_text)
    
    # Retrieve relevant chunks
    retrieved = search(query_vec, index, chunks, top_k=5)
    context = "\n".join([r[0] for r in retrieved])

    # Build RAG prompt
    prompt = f"""
    You are an assistant. Only answer using the provided context.
    If the context is insufficient, say "I don't know."

    Context:
    {context}

    Question:
    {query_text}
    """

    # Call Gemini model
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    return response.text
