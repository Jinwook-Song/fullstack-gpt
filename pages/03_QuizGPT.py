import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    if not os.path.exists("./.cache/quiz_files"):
        os.makedirs("./.cache/quiz_files")
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load_and_split(text_splitter=text_splitter)
    return documents


st.set_page_config(page_title="QuizGPT", page_icon="❓")

st.title("QuizGPT")

with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            label="Upload a file(.txt, .pdf, .docs)",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)

    else:
        topic = st.text_input(label="Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)  # type: ignore
            with st.status("Searching wikipedia..."):
                docs = retriever.get_relevant_documents(topic)
                st.write(docs)
