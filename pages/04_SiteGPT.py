from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st


st.set_page_config(page_title="SiteGPT", page_icon="📊")


llm = ChatOpenAI(temperature=0.1)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.
    If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    The score should be high if the answer is related to the user's question, and low otherwise.
    If there is no relevant content, the score is 0.
    Always provide scores with your answers
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"context": doc.page_content, "question": question}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are with Date, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f'{answer["answer"]}\nSource:{answer["source"]}\nDate{answer["date"]}\n'
        for answer in answers
    )

    return choose_chain.invoke(
        {"question": question, "answers": condensed},
    )


def parse_page(soup):
    """
    header와 footer를 제거한 contents 반환
    """
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")  # 줄바꿈
        .replace("\xa0", " ")  # 공백 문자
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        # filter_urls=[r"^(.*\/blog\/).*"],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5  # 너무 빠르면 차단 당할 수 있다 (default 2)
    docs = loader.load_and_split(text_splitter=text_splitter)
    vector_store = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
    return vector_store.as_retriever()


################################################################
st.markdown(
    """
# SiteGPT

Ask questions about the content of a website.

Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\\$"))  # type: ignore
