import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000/query"

# Set Streamlit page config
st.set_page_config(page_title="AI-Powered Knowledge Assistant", page_icon="🤖")

# Title
st.title("📚 AI-Powered Knowledge Assistant")

# Sidebar for file upload
st.sidebar.header("📂 Upload Documents")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Save uploaded file
if uploaded_file:
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")

# Chat-like UI
st.subheader("💬 Ask a Question")
user_query = st.text_input("Type your question:")

# Cache responses for performance
@st.cache_resource
def query_api(question):
    response = requests.post(API_URL, json={"query": question})
    return response.json().get("response", "No response available.")

# Send query to API
if st.button("Ask AI"):
    if user_query:
        with st.spinner("Thinking... 🤔"):
            answer = query_api(user_query)
        st.markdown(f"**🤖 AI Response:** {answer}")
    else:
        st.warning("Please enter a question.")
