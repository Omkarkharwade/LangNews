import os
import time
import pickle
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# ======================
# ✅ CONFIG
# ======================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="News Research Tool", page_icon="📰")
st.title("📰 News Research Tool (Gemini + FAISS)")
st.sidebar.header("🔗 Input News URLs")

# ======================
# URL Input
# ======================
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(1)]
urls = [u.strip() for u in urls if u.strip()]
process_url_clicked = st.sidebar.button("🚀 Process URLs")

file_path = "faiss_store_hugging.pkl"
main_placeholder = st.empty()

# ======================
# FAISS Vectorstore (loaded when needed for queries)
# ======================
vectorstore = None

# ======================
# Process URLs
# ======================
if process_url_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        print(f"🚀 Processing URLs: {urls}")
        main_placeholder.text("🧠 Loading data from URLs... please wait...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        print(f"📄 Loaded {len(data)} documents from URLs")

        # Split text into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )

        # 💡 RPM Delay Setup
        rpm = 15
        delay_per_operation = 60 / rpm  # 4 seconds per operation
        st.sidebar.write(f"⏳ Delay set to {delay_per_operation} seconds per URL")

        docs_all = []
        for idx, doc in enumerate(data):
            st.write(f"📄 Processing URL {idx + 1}/{len(data)}...")
            docs = text_splitter.split_documents([doc])
            docs_all.extend(docs)
            print(f"📄 Created {len(docs)} chunks from URL {idx + 1}")

            # Add time delay between URLs (15 RPM)
            if idx < len(data) - 1:
                st.info(f"⏸️ Waiting {delay_per_operation} seconds before next URL...")
                time.sleep(delay_per_operation)

        print(f"📊 Total chunks created: {len(docs_all)}")
        print("🧠 Creating embeddings with HuggingFace...")
        # Create embeddings (local HuggingFace model, no API limits)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create new FAISS vectorstore from current URL data only
        vectorstore = FAISS.from_documents(docs_all, embeddings)
        print("🆕 Created new FAISS index (previous data deleted)")

        # Save FAISS store (overwrites any existing file)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
        print(f"💾 FAISS index saved to {file_path}")

        main_placeholder.success("✅ FAISS index updated successfully!")

# ======================
# Lightweight LLM for Q&A
# ======================
try:
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')  # ⚡ fast + cheaper
except Exception as e:
    st.error(f"⚠️ Model init error: {e}")

# ======================
# User Query
# ======================
query = st.text_input("💬 Ask a question about the articles:")

if query:
    print(f"🤔 Processing question: {query}")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        print("📁 Loaded FAISS index from disk")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # limit retrieved chunks

        # Get and log retrieved chunks
        retrieved_docs = retriever.get_relevant_documents(query)
        print("🔍 Retrieved chunks for question:")
        for i, doc in enumerate(retrieved_docs):
            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            print(f"  Chunk {i+1}: {content_preview}")

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        with st.spinner("🤖 Thinking..."):
            result = chain.invoke({"question": query})

        st.subheader("🧠 Answer")
        st.write(result["answer"])

        sources = result["sources"]
        if sources:
            st.subheader("📚 Sources:")
            for src in sources.split("\n"):
                if src.strip():
                    st.write("🔗", src.strip())
    else:
        st.error("⚠️ Please process URLs first.")
