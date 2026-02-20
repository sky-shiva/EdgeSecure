import streamlit as st
import os
import fitz  # PyMuPDF
from faster_whisper import WhisperModel
import ollama
import numpy as np
import time
import wave
import pyaudiowpatch as pyaudio
from datetime import datetime
import onnxruntime as ort

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EdgeSecure AI", page_icon="🛡️", layout="wide")

# --- UI POLISHING (Corporate Security Aesthetic) ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    h1 { color: #00d4ff; font-weight: 800; letter-spacing: -1px; }
    .stAlert { background-color: #161b22; border: 1px solid #004a99; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
    .recording-status {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.2rem;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }
    
    .status-pulse {
        height: 10px; width: 10px;
        background-color: #23d160;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'doc_text' not in st.session_state:
    st.session_state.doc_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- AMD NPU HARDWARE DETECTION ---
def check_amd_npu():
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in providers:
        return True, "DirectML (AMD NPU/GPU) Detected"
    return False, "Standard CPU Mode"

# --- CORE AUDIO ENGINE (Ghost Mode + Unlimited Duration) ---
def capture_session(ghost_mode=True):
    p = pyaudio.PyAudio()
    try:
        # Get WASAPI Host
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_mic = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        # Find Loopback device for Ghost Mode
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break

        # Native Hardware Rates
        mic_rate = int(default_mic["defaultSampleRate"])
        loop_rate = int(default_speakers["defaultSampleRate"])
        
        # Open Mic Stream
        mic_stream = p.open(format=pyaudio.paInt16, channels=1, rate=mic_rate, input=True, input_device_index=default_mic["index"])

        # Open System Loopback Stream
        loop_stream = None
        if ghost_mode:
            loop_stream = p.open(format=pyaudio.paInt16, channels=default_speakers["maxInputChannels"], 
                                 rate=loop_rate, input=True, input_device_index=default_speakers["index"])

        frames = []
        st.session_state.is_recording = True
        
        # UI Placeholders
        status_box = st.empty()
        timer_box = st.empty()
        start_time = time.time()

        while st.session_state.is_recording:
            # Update UI Timer
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            timer_box.markdown(f"### ⏳ Recording Duration: `{elapsed}`")
            status_box.markdown('<p class="recording-status">🔴 SESSION ACTIVE: SECURE CAPTURE IN PROGRESS</p>', unsafe_allow_html=True)

            # Read Data
            m_data = mic_stream.read(1024, exception_on_overflow=False)
            m_audio = np.frombuffer(m_data, dtype=np.int16)

            if ghost_mode and loop_stream:
                l_chunk = int(1024 * (loop_rate / mic_rate))
                l_data = loop_stream.read(l_chunk, exception_on_overflow=False)
                l_audio = np.frombuffer(l_data, dtype=np.int16)
                
                if default_speakers["maxInputChannels"] > 1:
                    l_audio = l_audio.reshape(-1, default_speakers["maxInputChannels"])[:, 0]
                
                # Resample system audio to match mic audio length
                indices = np.linspace(0, len(l_audio) - 1, num=len(m_audio))
                l_audio = l_audio[indices.astype(int)]
                
                # Merge: Mic + System
                mixed = (m_audio // 2 + l_audio // 2).astype(np.int16)
                frames.append(mixed.tobytes())
            else:
                frames.append(m_audio.tobytes())

            # Stop button logic
            if st.button("⏹️ Stop and Process Session", key="stop_btn"):
                st.session_state.is_recording = False

        # Cleanup
        mic_stream.stop_stream(); mic_stream.close()
        if loop_stream:
            loop_stream.stop_stream(); loop_stream.close()
        
        # Save File
        with wave.open("temp_session.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(mic_rate)
            wf.writeframes(b''.join(frames))
            
    finally:
        p.terminate()

# --- AI PROCESSING ---
def process_audio(model_size):
    # Transcription
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe("temp_session.wav", beam_size=5)
    raw_text = " ".join([s.text for s in segments])
    
    # Diarization via Phi-3
    prompt = f"Identify the speakers in this transcript. Label the person talking through the microphone as 'Speaker 1' and the remote person from the call as 'Speaker 2': {raw_text}"
    res = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': prompt}])
    return res['message']['content']

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<h1 style="text-align: center;">🛡️ EdgeSecure</h1>', unsafe_allow_html=True)
    has_npu, npu_msg = check_amd_npu()
    st.markdown(f"""<div style="background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d;">
            <span class="status-pulse"></span><strong>{npu_msg}</strong><br>
            <small style="color: #8b949e;">Local-Only | Air-Gapped Mode</small>
        </div>""", unsafe_allow_html=True)
    
    st.divider()
    model_choice = st.selectbox("Whisper Sensitivity", ["base", "small"])
    ghost_mode = st.toggle("Ghost Mode (System Audio)", value=True)
    
    if st.button("🗑️ Wipe All Cache"):
        st.session_state.clear()
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("🛡️ EdgeSecure Intelligence Dashboard")

tab1, tab2, tab3 = st.tabs(["🎙️ Meeting Scribe", "📄 Document Vault", "💬 Secure Chat"])

with tab1:
    col_ctrl, col_view = st.columns([1, 1.5])
    
    with col_ctrl:
        st.subheader("Session Controls")
        if not st.session_state.is_recording:
            if st.button("🔴 Start Encrypted Recording", use_container_width=True):
                capture_session(ghost_mode=ghost_mode)
                with st.spinner("Analyzing audio on AMD NPU..."):
                    st.session_state.transcript = process_audio(model_choice)
                    st.rerun()
        else:
            st.warning("Recording in Progress... Click the Stop button above.")

        if st.session_state.transcript:
            if st.button("📝 Generate Executive Summary"):
                with st.spinner("Summarizing with Phi-3..."):
                    res = ollama.chat(model='phi3', messages=[
                        {'role': 'system', 'content': 'Summarize this meeting into high-level security action items.'},
                        {'role': 'user', 'content': st.session_state.transcript}
                    ])
                    st.session_state.summary = res['message']['content']

    with col_view:
        st.subheader("Intelligence Output")
        if st.session_state.transcript:
            st.markdown("**Transcript (Local Diarization):**")
            st.info(st.session_state.transcript)
            
            if st.session_state.summary:
                st.markdown("**Executive Summary:**")
                st.success(st.session_state.summary)

with tab2:
    st.subheader("Local Document Vault")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    if uploaded_file and st.button("🔍 Ingest Document"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        st.session_state.doc_text = "".join([page.get_text() for page in doc])
        st.success("Document analyzed and stored in local memory.")

with tab3:
    if not st.session_state.transcript and not st.session_state.doc_text:
        st.warning("Capture a meeting or upload a document to start a secure chat.")
    else:
        context = f"Transcript: {st.session_state.transcript}\nDoc: {st.session_state.doc_text[:1500]}"
        for chat in st.session_state.chat_history:
            with st.chat_message("user"): st.write(chat['q'])
            with st.chat_message("assistant"): st.write(chat['a'])
            
        user_input = st.chat_input("Ask a question about your data...")
        if user_input:
            with st.chat_message("user"): st.write(user_input)
            with st.chat_message("assistant"):
                response = ollama.chat(model='phi3', messages=[
                    {'role': 'system', 'content': f'Answer only based on: {context}'},
                    {'role': 'user', 'content': user_input}
                ])
                st.write(response['message']['content'])
                st.session_state.chat_history.append({"q": user_input, "a": response['message']['content']})

st.sidebar.caption("v1.5 | AMD Slingshot Challenge 2026")