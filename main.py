import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# UI setup
st.title("📰 Financial News Research Tool")
st.sidebar.title("Enter News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("🔄 Process URLs")
vectorstore_dir = "faiss_index"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Process URLs and save FAISS index
if process_url_clicked:
    urls = [url.strip() for url in urls if url.strip()]
    if not urls:
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        try:
            loader = WebBaseLoader(urls)
            main_placeholder.text("🔄 Loading articles...")
            data = loader.load()

            if not data:
                st.error("❌ Failed to load content from URLs. Try different links.")
            else:
                main_placeholder.text("✂️ Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)

                if not docs:
                    st.error("❌ No documents created after splitting.")
                else:
                    main_placeholder.text("🔍 Creating embeddings and saving index...")
                    embeddings = OpenAIEmbeddings()
                    vectorstore_openai = FAISS.from_documents(docs, embeddings)
                    vectorstore_openai.save_local(vectorstore_dir)

                    main_placeholder.success("✅ Index built and saved successfully!")

        except Exception as e:
            st.error(f"❌ Error processing URLs: {e}")

# Handle questions
query = main_placeholder.text_input("💬 Ask a question about the articles:")

if query:
    if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
        try:
            vectorstore = FAISS.load_local(
                vectorstore_dir,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)

            st.header("🧠 Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("📎 Sources:")
                for source in sources.split("\n"):
                    st.write(source)

        except Exception as e:
            st.error(f"❌ Error running query: {e}")
    else:
        st.warning("⚠️ Please process URLs first to build the index.")
