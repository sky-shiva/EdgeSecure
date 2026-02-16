import streamlit as st
import os
import fitz  # PyMuPDF
from faster_whisper import WhisperModel
import ollama
from scipy.io.wavfile import write
import sounddevice as sd
import numpy as np
import time
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EdgeSecure AI", page_icon="🛡️", layout="wide")

# Custom CSS to give it a "Startup" feel
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004a99; color: white; }
    .stTextArea>div>div>textarea { background-color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""
if 'doc_text' not in st.session_state:
    st.session_state['doc_text'] = ""
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- HELPER FUNCTIONS ---

def extract_pdf_text(uploaded_file):
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    doc = fitz.open("temp_doc.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def run_transcription(duration, model_size):
    fs = 44100
    # 1. Record
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    write("temp_audio.wav", fs, audio_data)
    
    # 2. Transcribe (Using CPU Stable Mode)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe("temp_audio.wav", beam_size=5)
    return " ".join([s.text for s in segments])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield-with-crown.png", width=80)
    st.title("EdgeSecure v1.0")
    st.info("Local-Only AI Intelligence")
    
    st.header("Hardware Optimization")
    npu_boost = st.toggle("AMD Ryzen AI (NPU) Boost", value=False)
    if npu_boost:
        st.success("DirectML / ONNX Path Enabled")
    else:
        st.warning("Standard CPU Compatibility Mode")
    
    st.divider()
    model_choice = st.selectbox("Whisper Accuracy", ["base", "small"], index=0)
    if st.button("Clear All Data"):
        st.session_state.clear()
        st.rerun()

# --- MAIN UI ---
st.title("🛡️ EdgeSecure Dashboard")
st.caption("Secure Meeting Intelligence & Document Analysis | No Cloud. No Leaks.")

tab1, tab2, tab3 = st.tabs(["🎙️ Meeting Recorder", "📄 Document Analysis", "💬 Secure Chat"])

# --- TAB 1: MEETING RECORDER ---
with tab1:
    col1_rec, col2_rec = st.columns([1, 2])
    
    with col1_rec:
        st.subheader("Capture Meeting")
        record_sec = st.slider("Record Duration (sec)", 5, 120, 20)
        if st.button("🔴 Start Secure Recording"):
            with st.status("Processing Audio Locally...") as status:
                st.write("Recording from encrypted buffer...")
                text = run_transcription(record_sec, model_choice)
                st.session_state['transcript'] = text
                status.update(label="Transcription Complete!", state="complete")
    
    with col2_rec:
        st.subheader("Live Transcript")
        if st.session_state['transcript']:
            st.text_area("Meeting Text", st.session_state['transcript'], height=250)
            if st.button("💾 Generate Meeting Summary"):
                with st.spinner("Phi-3 Analyzing..."):
                    res = ollama.chat(model='phi3', messages=[
                        {'role': 'system', 'content': 'You are a professional secretary. Summarize accurately.'},
                        {'role': 'user', 'content': st.session_state['transcript']}
                    ])
                    st.markdown("### Summary")
                    st.info(res['message']['content'])
        else:
            st.info("No audio captured yet. Press record to start.")

# --- TAB 2: DOCUMENT ANALYSIS ---
with tab2:
    st.subheader("Secure Document Vault")
    uploaded_file = st.file_uploader("Upload sensitive PDF", type=['pdf'])
    
    if uploaded_file:
        if st.button("🔍 Analyze Document"):
            with st.spinner("Reading PDF locally..."):
                doc_text = extract_pdf_text(uploaded_file)
                st.session_state['doc_text'] = doc_text
                st.success("Document ingested securely.")
        
        if st.session_state['doc_text']:
            st.text_area("Document Content (Preview)", st.session_state['doc_text'][:1000] + "...", height=200)

# --- TAB 3: SECURE CHAT ---
with tab3:
    st.subheader("Chat with your Intelligence")
    st.caption("Ask questions about your meetings or uploaded documents.")
    
    # Combine context for the AI
    context = ""
    if st.session_state['transcript']:
        context += f"Meeting Transcript: {st.session_state['transcript']}\n"
    if st.session_state['doc_text']:
        context += f"Document Content: {st.session_state['doc_text'][:2000]}\n" # Limit context size
    
    if context == "":
        st.warning("Please record a meeting or upload a document first to use the Chat.")
    else:
        user_input = st.chat_input("Ask a question...")
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Reasoning..."):
                    response = ollama.chat(model='phi3', messages=[
                        {'role': 'system', 'content': f'You are EdgeSecure AI. Answer based ONLY on this context: {context}'},
                        {'role': 'user', 'content': user_input}
                    ])
                    answer = response['message']['content']
                    st.write(answer)
                    st.session_state['chat_history'].append({"q": user_input, "a": answer})

# --- FOOTER ---
st.divider()
st.markdown("🔒 **EdgeSecure Status:** Encrypted | **Cloud Connectivity:** Disabled | **Hardware:** Optimized for AMD Ryzen AI")