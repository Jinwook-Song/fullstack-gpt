from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st

st.set_page_config(page_title="DocumentGPT", page_icon="📃")


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=cached_embeddings,
    )
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
Welcom!

Use the chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader(
    label="Upload a file(.txt, .pdf, .docs)",
    type=["pdf", "txt", "docx"],
)

if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    s
