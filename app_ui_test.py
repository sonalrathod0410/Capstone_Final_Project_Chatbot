import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000/query"
UPLOAD_URL = "http://127.0.0.1:8000/upload_pdf"

st.set_page_config(page_title="AI-Powered Knowledge Assistant", page_icon="🤖")
st.title("📚 AI-Powered Knowledge Assistant")

st.sidebar.header("📂 Upload Documents")

uploaded_file = st.sidebar.file_uploader("Upload a document (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading and processing... 📄"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(UPLOAD_URL, files=files)
        if response.status_code == 200:
            st.sidebar.success("✅ PDF uploaded and indexed successfully!")
        else:
            st.sidebar.error("❌ Failed to upload PDF.")

st.subheader("💬 Ask a Question")
user_query = st.text_input("Type your question:")

@st.cache_resource
def query_api(question):
    response = requests.post(API_URL, json={"query": question})
    return response.json().get("response", "No response available.")

if st.button("Ask AI"):
    if user_query:
        with st.spinner("Thinking... 🤔"):
            answer = query_api(user_query)
        st.markdown(f"**🤖 AI Response:** {answer}")
    else:
        st.warning("Please enter a question.")
