from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os

# Initialize FastAPI app
app = FastAPI()

# Load LLM model
llm = Ollama(model="mistral")

# Load embeddings model (optimized for speed)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB storage path
CHROMA_PATH = "chroma_db"

# Load ChromaDB
vectorstore = Chroma(collection_name="documents", persist_directory=CHROMA_PATH, embedding_function=embedding_model)

# Use better retrieval strategy
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Define QA chain
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
