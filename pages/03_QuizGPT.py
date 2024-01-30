import os
from typing import List
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema import Document
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


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


st.set_page_config(page_title="QuizGPT", page_icon="❓")

st.title("QuizGPT")

with st.sidebar:
    docs = None
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
            retriever = WikipediaRetriever(
                top_k_results=5,
                lang="ko",
            )  # type: ignore
            with st.status("Searching wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
Welcom to QuizGPT.

I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

Get started by uploading a file or searching on Wikipedia in the sidebar.

"""
    )
else:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    Use (o) to signal the correct answer.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    Your turn!

    Context: {context}
""",
            )
        ]
    )

    chain = {"context": format_docs} | prompt | llm

    start = st.button("Generate Quiz")

    if start:
        chain.invoke(docs)
