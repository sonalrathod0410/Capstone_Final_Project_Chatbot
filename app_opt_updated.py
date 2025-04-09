from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
import streamlit as st

# Initialize FastAPI app
app = FastAPI()

# Cache the embeddings model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ChromaDB storage path
CHROMA_PATH = "chroma_db"

# Load ChromaDB
vectorstore = Chroma(collection_name="documents", persist_directory=CHROMA_PATH, embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Define QA chain
llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# API request model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def search_and_generate_response(request: QueryRequest):
    """Retrieve documents and generate AI-powered response"""
    response = qa_chain.invoke(request.query)
    return {"query": request.query, "response": response["result"]}

@app.get("/")
def home():
    return {"message": "Mistral AI-powered search API is running!"}