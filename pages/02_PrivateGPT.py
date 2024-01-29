import os
from typing import List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st

st.set_page_config(page_title="PrivateGPT", page_icon="🔐")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner=True)
def embed_file(file):
    file_content = file.read()
    if not os.path.exists("./.cache/private_files"):
        os.makedirs("./.cache/private_files")
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    if not os.path.exists("./.cache/private_embeddings"):
        os.makedirs("./.cache/private_embeddings")
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OllamaEmbeddings(model="mistral:latest")
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


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the following context and not training data.
If you don't know the answer just say you don't know.
DON'T make anyting up.

Context: {context}
Question: {question}
         """,
)


st.title("DocumentGPT")

st.markdown(
    """
Welcom!

Use the chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        label="Upload a file(.txt, .pdf, .docs)",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about this file")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)


else:
    st.session_state["messages"] = []
