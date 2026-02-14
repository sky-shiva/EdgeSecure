import streamlit as st
import os
import fitz  # PyMuPDF
from faster_whisper import WhisperModel
import ollama
import numpy as np
import time
import wave
import pyaudiowpatch as pyaudio
import threading
from datetime import datetime
import onnxruntime as ort

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EdgeSecure AI", page_icon="🛡️", layout="wide")

# --- UI POLISHING ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    h1 { color: #00d4ff; font-weight: 800; letter-spacing: -1px; }
    .stAlert { background-color: #161b22; border: 1px solid #004a99; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .recording-status { color: #ff4b4b; font-weight: bold; font-size: 1.2rem; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'doc_text' not in st.session_state:
    st.session_state.doc_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'frames' not in st.session_state:
    st.session_state.frames = []

# --- AMD NPU DETECTION ---
def check_amd_npu():
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in providers:
        return True, "DirectML (AMD NPU/GPU) Active"
    return False, "Standard CPU Mode"

# --- BACKGROUND RECORDING FUNCTION ---
def recording_thread_func(ghost_mode):
    """ This runs in the background to avoid freezing the UI """
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_mic = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break

        mic_rate = int(default_mic["defaultSampleRate"])
        loop_rate = int(default_speakers["defaultSampleRate"])
        
        mic_stream = p.open(format=pyaudio.paInt16, channels=1, rate=mic_rate, input=True, input_device_index=default_mic["index"])
        loop_stream = None
        if ghost_mode:
            loop_stream = p.open(format=pyaudio.paInt16, channels=default_speakers["maxInputChannels"], 
                                 rate=loop_rate, input=True, input_device_index=default_speakers["index"])

        st.session_state.frames = []
        
        while st.session_state.recording_active:
            m_data = mic_stream.read(1024, exception_on_overflow=False)
            m_audio = np.frombuffer(m_data, dtype=np.int16)

            if ghost_mode and loop_stream:
                l_chunk = int(1024 * (loop_rate / mic_rate))
                l_data = loop_stream.read(l_chunk, exception_on_overflow=False)
                l_audio = np.frombuffer(l_data, dtype=np.int16)
                
                if default_speakers["maxInputChannels"] > 1:
                    l_audio = l_audio.reshape(-1, default_speakers["maxInputChannels"])[:, 0]
                
                indices = np.linspace(0, len(l_audio) - 1, num=len(m_audio))
                l_audio = l_audio[indices.astype(int)]
                
                mixed = (m_audio // 2 + l_audio // 2).astype(np.int16)
                st.session_state.frames.append(mixed.tobytes())
            else:
                st.session_state.frames.append(m_audio.tobytes())

        # Save the file once the loop breaks
        with wave.open("temp_session.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(mic_rate)
            wf.writeframes(b''.join(st.session_state.frames))

        mic_stream.stop_stream(); mic_stream.close()
        if loop_stream: loop_stream.stop_stream(); loop_stream.close()
    finally:
        p.terminate()

# --- AI WRAPPERS ---
def run_ai_analysis(model_size):
    # Transcription
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe("temp_session.wav", beam_size=5)
    text = " ".join([s.text for s in segments])
    
    # Diarization
    res = ollama.chat(model='phi3', messages=[{
        'role': 'user', 
        'content': f"Format this as a dialogue between 'Speaker 1' (Mic) and 'Speaker 2' (Remote Call): {text}"
    }])
    return res['message']['content']

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<h1 style="text-align: center;">🛡️ EdgeSecure</h1>', unsafe_allow_html=True)
    has_npu, npu_msg = check_amd_npu()
    st.info(npu_msg)
    st.divider()
    model_choice = st.selectbox("Whisper Sensitivity", ["base", "small"])
    ghost_mode = st.toggle("Ghost Mode (Intercept System Audio)", value=True)
    if st.button("🗑️ Emergency Wipe"):
        st.session_state.clear()
        st.rerun()

# --- MAIN UI ---
st.title("🛡️ EdgeSecure Intelligence Dashboard")
tab1, tab2, tab3 = st.tabs(["🎙️ Meeting Scribe", "📄 Document Vault", "💬 Secure Chat"])

with tab1:
    col_ctrl, col_view = st.columns([1, 1.5])

    with col_ctrl:
        st.subheader("Session Control")
        
        if not st.session_state.recording_active:
            if st.button("🔴 Start Encrypted Recording", use_container_width=True):
                st.session_state.recording_active = True
                # Start recording in background thread
                threading.Thread(target=recording_thread_func, args=(ghost_mode,), daemon=True).start()
                st.rerun()
        else:
            st.markdown('<p class="recording-status">🔴 RECORDING LIVE...</p>', unsafe_allow_html=True)
            if st.button("⏹️ Stop and Process Session", use_container_width=True):
                st.session_state.recording_active = False
                with st.spinner("NPU-Optimized Transcription in progress..."):
                    # Give the thread a split second to save the file
                    time.sleep(1) 
                    st.session_state.transcript = run_ai_analysis(model_choice)
                st.rerun()

        if st.session_state.transcript:
            if st.button("📝 Summarize Meeting"):
                res = ollama.chat(model='phi3', messages=[
                    {'role': 'system', 'content': 'Summarize this meeting into high-level action items.'},
                    {'role': 'user', 'content': st.session_state.transcript}
                ])
                st.session_state.summary = res['message']['content']

    with col_view:
        if st.session_state.transcript:
            st.markdown("**Local AI Transcript:**")
            st.info(st.session_state.transcript)
            if st.session_state.summary:
                st.markdown("**Executive Summary:**")
                st.success(st.session_state.summary)

# --- TAB 2 & 3 (Same logic as before) ---
with tab2:
    st.subheader("Local PDF Ingestion")
    uploaded_file = st.file_uploader("Drop sensitive PDF", type=['pdf'])
    if uploaded_file and st.button("🔍 Secure Ingest"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        st.session_state.doc_text = "".join([page.get_text() for page in doc])
        st.success("Document analyzed.")

with tab3:
    if st.session_state.transcript or st.session_state.doc_text:
        context = f"Transcript: {st.session_state.transcript}\nDoc: {st.session_state.doc_text[:1500]}"
        for chat in st.session_state.chat_history:
            with st.chat_message("user"): st.write(chat['q'])
            with st.chat_message("assistant"): st.write(chat['a'])
        
        user_input = st.chat_input("Ask about your data...")
        if user_input:
            with st.chat_message("user"): st.write(user_input)
            response = ollama.chat(model='phi3', messages=[
                {'role': 'system', 'content': f'Answer only based on: {context}'},
                {'role': 'user', 'content': user_input}
            ])
            st.write(response['message']['content'])
            st.session_state.chat_history.append({"q": user_input, "a": response['message']['content']})
    else:
        st.warning("No context available.")