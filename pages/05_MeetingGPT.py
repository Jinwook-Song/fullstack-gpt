from glob import glob
import math
import os
import subprocess
import openai
from pydub import AudioSegment
import streamlit as st

st.set_page_config(page_title="MeetingGPT", page_icon="🎥")

has_transcript = os.path.exists("./.cache/videos/llm_intro.txt")


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
    with st.status("Loading video..."):
        video_content = video.read()
        video_path = f"./.cache/videos/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)
    with st.status("Splitting audio segments..."):
        split_audio(audio_path, chunks_folder)
    with st.status("Transcribing audio..."):
        transcribe_chunks(chunks_folder, transcript_path)
