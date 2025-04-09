from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
import fitz  # PyMuPDF for PDF extraction
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

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and store PDF content into ChromaDB"""
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    chunked_texts = chunk_text(text)

    # Store chunks in ChromaDB
    for i, chunk in enumerate(chunked_texts):
        vectorstore.add_texts([chunk], metadatas=[{"source": file.filename, "chunk_id": i}])

    return {"message": f"PDF {file.filename} uploaded and indexed successfully"}

def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    return splitter.split_text(text)

@app.get("/")
def home():
    return {"message": "Mistral AI-powered search API is running!"}
