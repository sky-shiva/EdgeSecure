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

# --- 1. SETTINGS & PATHS ----
st.set_page_config(page_title="EdgeSecure AI", page_icon="🛡️", layout="wide")
ABS_FILE_PATH = os.path.abspath("temp_session.wav")

# --- 2. CSS STYLING ---
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

# --- 3. INITIALIZE SESSION STATE ---
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
if 'file_ready' not in st.session_state:
    st.session_state.file_ready = False

# --- 4. HARDWARE & AI FUNCTIONS ---
def check_amd_npu():
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in providers:
        return True, "DirectML (AMD Ryzen AI) Active"
    return False, "Standard CPU Mode"

def recording_background_worker(ghost_mode):
    """ Background thread to capture audio without freezing UI """
    p = pyaudio.PyAudio()
    try:
        # WASAPI Loopback Config
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_mic = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if ghost_mode and not default_speakers["isLoopbackDevice"]:
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

        frames = []
        
        while st.session_state.recording_active:
            m_data = mic_stream.read(1024, exception_on_overflow=False)
            m_audio = np.frombuffer(m_data, dtype=np.int16)

            if ghost_mode and loop_stream:
                l_chunk = int(1024 * (loop_rate / mic_rate))
                l_data = loop_stream.read(l_chunk, exception_on_overflow=False)
                l_audio = np.frombuffer(l_data, dtype=np.int16)
                
                # Resample system audio to match mic
                if default_speakers["maxInputChannels"] > 1:
                    l_audio = l_audio.reshape(-1, default_speakers["maxInputChannels"])[:, 0]
                indices = np.linspace(0, len(l_audio) - 1, num=len(m_audio))
                l_audio = l_audio[indices.astype(int)]
                
                mixed = (m_audio // 2 + l_audio // 2).astype(np.int16)
                frames.append(mixed.tobytes())
            else:
                frames.append(m_audio.tobytes())

        # WRITE FILE
        with wave.open(ABS_FILE_PATH, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(mic_rate)
            wf.writeframes(b''.join(frames))
        
        st.session_state.file_ready = True
        
        mic_stream.stop_stream(); mic_stream.close()
        if loop_stream: loop_stream.stop_stream(); loop_stream.close()
    finally:
        p.terminate()

def run_ai_analysis(model_size):
    """ AI Pipeline: Whisper -> Phi-3 Diarization """
    # Transcription
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(ABS_FILE_PATH, beam_size=5)
    raw_text = " ".join([s.text for s in segments])
    
    # Diarization Logic
    prompt = f"Convert this text into a dialogue. 'Speaker 1' is the local mic, 'Speaker 2' is the remote caller: {raw_text}"
    res = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': prompt}])
    return res['message']['content']

# --- 5. SIDEBAR UI ---
with st.sidebar:
    st.markdown('<h1 style="text-align: center;">🛡️ EdgeSecure</h1>', unsafe_allow_html=True)
    has_npu, npu_msg = check_amd_npu()
    st.success(f"NPU Status: {npu_msg}")
    st.divider()
    model_choice = st.selectbox("Whisper Sensitivity", ["base", "small"])
    ghost_mode = st.toggle("Ghost Mode (Intercept Meeting Audio)", value=True)
    if st.button("🗑️ Wipe Session Cache"):
        st.session_state.clear()
        if os.path.exists(ABS_FILE_PATH): os.remove(ABS_FILE_PATH)
        st.rerun()

# --- 6. MAIN UI ---
st.title("🛡️ Intelligence Control Center")
t1, t2, t3 = st.tabs(["🎙️ Meeting Scribe", "📄 Vault", "💬 Secure Chat"])

with t1:
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("Ghost Mode Capture")
        
        if not st.session_state.recording_active:
            if st.button("🔴 Start Live Recording", use_container_width=True):
                st.session_state.recording_active = True
                st.session_state.file_ready = False
                threading.Thread(target=recording_background_worker, args=(ghost_mode,), daemon=True).start()
                st.rerun()
        else:
            st.markdown('<p class="recording-status">🔴 RECORDING ACTIVE...</p>', unsafe_allow_html=True)
            if st.button("⏹️ Stop & Transcribe", use_container_width=True):
                st.session_state.recording_active = False
                # Block until file is ready
                with st.spinner("Closing Secure Stream & Saving..."):
                    while not st.session_state.file_ready:
                        time.sleep(0.1)
                
                with st.spinner("AI Diarization (NPU Optimized)..."):
                    st.session_state.transcript = run_ai_analysis(model_choice)
                st.rerun()

        if st.session_state.transcript:
            if st.button("📋 Generate Summary"):
                res = ollama.chat(model='phi3', messages=[
                    {'role': 'system', 'content': 'Provide a concise bulleted summary of action items.'},
                    {'role': 'user', 'content': st.session_state.transcript}
                ])
                st.session_state.summary = res['message']['content']

    with c2:
        if st.session_state.transcript:
            st.markdown("**Transcript (with Local Diarization):**")
            st.info(st.session_state.transcript)
            if st.session_state.summary:
                st.markdown("**Executive Summary:**")
                st.success(st.session_state.summary)

with t2:
    st.subheader("Local PDF Vault")
    up = st.file_uploader("Upload Sensitive Files", type=['pdf'])
    if up and st.button("🔍 Ingest PDF"):
        doc = fitz.open(stream=up.read(), filetype="pdf")
        st.session_state.doc_text = "".join([p.get_text() for p in doc])
        st.success("Document analyzed and stored in RAM.")

with t3:
    if st.session_state.transcript or st.session_state.doc_text:
        context = f"Transcript: {st.session_state.transcript}\nDoc: {st.session_state.doc_text[:1500]}"
        for chat in st.session_state.chat_history:
            with st.chat_message("user"): st.write(chat['q'])
            with st.chat_message("assistant"): st.write(chat['a'])
        
        u_in = st.chat_input("Query local data...")
        if u_in:
            with st.chat_message("user"): st.write(u_in)
            ans = ollama.chat(model='phi3', messages=[
                {'role': 'system', 'content': f'Answer based only on: {context}'},
                {'role': 'user', 'content': u_in}
            ])
            st.session_state.chat_history.append({"q": u_in, "a": ans['message']['content']})
            st.rerun()
    else:
        st.warning("No context available. Capture audio or upload a PDF first.")