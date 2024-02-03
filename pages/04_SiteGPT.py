from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


st.set_page_config(page_title="SiteGPT", page_icon="📊")


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
        filter_urls=[r"^(.*\/blog\/).*"],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5  # 너무 빠르면 차단 당할 수 있다 (default 2)
    docs = loader.load_and_split(text_splitter=text_splitter)
    return docs


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
        docs = load_website(url)
        st.write(docs)
