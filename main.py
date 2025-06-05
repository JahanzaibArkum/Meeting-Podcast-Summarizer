import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import tempfile
from moviepy import VideoFileClip

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Custom CSS
st.markdown("""
<style>
    /* Full page background */
    html, body, [class*="st"] {
        background-color: #ADD8E6 !important;  /* Light blue background */
        color: #1A1A1A;  /* Dark text */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Headings */
    h1, h2, h3 {
        color: #0A3D62;  /* Darker blue for headings */
    }

    /* Buttons */
    .stButton > button {
        color: #FFFFFF;
        background-color: #0A3D62;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #064273;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #B0E0E6;
        border: 1px solid #0A3D62;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Audio player */
    .stAudio {
        background-color: #B0E0E6;
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #FFFFFF;
        color: #1A1A1A;
        border: 1px solid #0A3D62;
        border-radius: 8px;
    }

    /* Markdown sections (optional) */
    .stMarkdown, .stTextArea {
        background: #B0E0E6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)




def extract_audio_from_video(video_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(temp_audio.name, codec="libmp3lame")
        return temp_audio.name

def transcribe_with_groq(audio_path):
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), file.read()),
            model="whisper-large-v3",
            response_format="json"
        )
    return transcription.text

    

    
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)  # e.g. 'en', 'ur', etc.
    except:
        return 'en'  # default to English if detection fails


st.title("🎙️ Meeting/Podcast Summarizer")

uploaded_file = st.file_uploader("Choose an audio or video file", type=["wav", "mp3", "m4a", "mp4", "mov", "avi"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    if uploaded_file.type.startswith("video"):
        st.video(file_bytes)
    else:
        st.audio(file_bytes)
    
    if st.button("🎬 Transcribe & Summarize"):
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        # Extract audio if it's a video
        if uploaded_file.type.startswith("video"):
            audio_path = extract_audio_from_video(temp_file_path)
            os.unlink(temp_file_path)
            temp_file_path = audio_path

        with st.spinner("Transcribing with Groq Whisper..."):
            transcription = transcribe_with_groq(temp_file_path)
        
        st.subheader("📝 Transcription:")
        st.text_area("", transcription, height=300)
        
        os.unlink(temp_file_path)

        with st.spinner("Summarizing with Groq LLM..."):
            try:
                detected_lang = detect_language(transcription)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                        {"role": "user", "content": f"Please summarize the following transcription in its original language (detected language: {detected_lang}):\n\n{transcription}"}
                    ],
                    temperature=0.5,
                )
                summary = response.choices[0].message.content
                st.subheader("📋 Summary:")
                st.markdown(summary)
            except Exception as e:
                st.error(f"❌ Error during summarization: {str(e)}")

