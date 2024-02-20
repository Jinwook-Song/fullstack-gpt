from glob import glob
import math
import os
import subprocess
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from pydub import AudioSegment
import streamlit as st

st.set_page_config(page_title="MeetingGPT", page_icon="🎥")


llm = ChatOpenAI(temperature=0.1)

transcribed_file_path = "./.cache/videos/llm_intro.txt"
has_transcript = os.path.exists(transcribed_file_path)


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = sorted(glob(f"{chunk_folder}/*.mp3"))

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    command = [
        "ffmpeg",
        "-y",  # always yes(override)
        "-i",
        video_path,
        "-vn",
        video_path.replace("mp4", "mp3"),
    ]
    subprocess.run(command)


@st.cache_data()
def split_audio(audio_path, chunks_folder, chunk_size=10):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{str(i).zfill(2)}.mp3", format="mp3")


################################################################
st.markdown(
    """
# MeetingGPT

Welcome to MeetingGPT, upload a video and I will give you a transcript,
a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "avi", "mkv", "mov"])

if video:
    chunks_folder = "./.cache/videos/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/videos/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Splitting audio segments...")
        split_audio(audio_path, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qna_tab = st.tabs(
        [
            "Transcirpt",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as f:
            st.write(f.read())

    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader(transcript_path)
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=text_splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """,
            )
            first_summary_chain = first_summary_prompt | llm | StrOutputParser()
            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarzing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    st.write(f"[Document{i+1}]\n{summary}")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )

            st.write(f"[Document{len(docs)}]\n{summary}")
