import streamlit as st
from langchain.prompts import PromptTemplate

st.write("hello")

st.write([1, 2, 3, 4])

st.write({"x": 1})

st.write(PromptTemplate)

st.selectbox(
    label="Choose your model",
    options=(
        "GPT-3.5",
        "GPT-4",
    ),
)

p = PromptTemplate.from_template("xxx")

st.write(p)

# magic
p
