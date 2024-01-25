from operator import itemgetter
from typing import List
import streamlit as st

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema import Document, messages_from_dict, messages_to_dict
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

import json
import os

st.set_page_config(page_title="DocumentGPT", page_icon="📃")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory_llm = ChatOpenAI(temperature=0.1)

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm,
        max_token_limit=120,
        memory_key="chat_history",
        return_messages=True,
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the question using ONLY the following context.
If you don't know the answer just say you don't know.
DON'T make anyting up.

Context: {context}

And you will get about summaried context of previous chat.
If it's empty you don't have to care Previous-chat-context: {chat_history}
         """,
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    if not os.path.exists("./.cache/files"):
        os.makedirs("./.cache/files")
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


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def save_memory(input, output):
    st.session_state["chat_history"].append({"input": input, "output": output})


def save_memory_on_file(memory_file_path):
    print("work save memory on file")
    history = st.session_state["memory"].chat_memory.messages
    history = messages_to_dict(history)

    with open(memory_file_path, "w") as f:
        json.dump(history, f)


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


def restore_memory():
    print("work restore memory")
    for history in st.session_state["chat_history"]:
        st.session_state["memory"].save_context(
            {"input": history["input"]}, {"output": history["output"]}
        )


def invoke_chain(message):
    # invoke the chain
    result = chain.invoke(message)
    # save the interaction in the memory
    save_memory(message, result.content)


@st.cache_data(show_spinner="Loading memory from file...")
def load_memory_from_file(memory_file_path):
    print("work load memory from file")
    loaded_message = load_json(memory_file_path)
    history = messages_from_dict(loaded_message)
    st.session_state["memory"].chat_memory.messages = history


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
    memory_checkbox = None
    memory_file_path = "./.cache/chat_memory/memory.json"
    if os.path.exists(memory_file_path):
        memory_checkbox = st.checkbox(
            "Do you want to keep your previous chat?", value=True
        )
        if memory_checkbox:
            load_memory_from_file(memory_file_path)

if file:
    retriever = embed_file(file)
    send_message("I'm ready Ask away!", "ai", save=False)
    restore_memory()
    paint_history()
    message = st.chat_input("Ask anything about this file")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough.assign(
                chat_history=RunnableLambda(
                    st.session_state["memory"].load_memory_variables
                )
                | itemgetter("chat_history")
            )
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            invoke_chain(message)

        if len(st.session_state["memory"].chat_memory.messages) != 0:
            save_memory_on_file(memory_file_path=memory_file_path)


else:
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []
