# src/embedder.py
import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def embed_chunks(chunks):
    vectors = embedding_model.embed_documents(chunks)
    return np.array(vectors, dtype="float32")

def embed_query(query):
    vector = embedding_model.embed_query(query)
    return np.array(vector, dtype="float32")
