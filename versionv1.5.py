import streamlit as st
import os
import sys
import fitz  # PyMuPDF
import numpy as np
import time
import wave
import threading
from datetime import datetime
import tempfile

# Try to import audio libraries (with graceful fallback)
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import ollama
except ImportError:
    ollama = None

# --- 1. SETTINGS & PATHS ----
st.set_page_config(page_title="EdgeSecure AI", page_icon="🛡️", layout="wide")
TEMP_DIR = tempfile.gettempdir()
WAV_FILE = os.path.join(TEMP_DIR, "edgesecure_session.wav")

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    h1 { color: #00d4ff; font-weight: 800; letter-spacing: -1px; }
    .stAlert { background-color: #161b22; border: 1px solid #004a99; border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    .recording-status { color: #ff4b4b; font-weight: bold; font-size: 1.2rem; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
    .status-box { padding: 15px; border-radius: 8px; margin: 10px 0; }
    .status-ok { background-color: #1a472a; border-left: 4px solid #0caf00; }
    .status-error { background-color: #3a1a1a; border-left: 4px solid #ff4444; }
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
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 16000

# --- 4. DEPENDENCY CHECK ---
def check_dependencies():
    """Check which dependencies are installed"""
    deps = {
        'sounddevice': sd is not None,
        'faster-whisper': WhisperModel is not None,
        'ollama': ollama is not None,
        'PyMuPDF': True,  # Already imported
        'streamlit': True,
    }
    return deps

def get_dependency_status():
    """Return user-friendly status"""
    deps = check_dependencies()
    status_html = "<div class='status-box'>"
    
    for dep, available in deps.items():
        icon = "✅" if available else "❌"
        status = "Available" if available else "Missing"
        status_html += f"{icon} {dep}: {status}<br>"
    
    status_html += "</div>"
    return status_html, deps

# --- 5. AUDIO RECORDING (SIMPLIFIED) ---
def record_audio_simple(duration=60):
    """Simple audio recording using sounddevice (cross-platform)"""
    if sd is None:
        return None, "sounddevice not installed. Run: pip install sounddevice"
    
    try:
        sample_rate = 16000
        st.session_state.sample_rate = sample_rate
        
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        
        # Save to WAV
        with wave.open(WAV_FILE, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return WAV_FILE, None
    except Exception as e:
        return None, str(e)

# --- 6. AI ANALYSIS ---
def transcribe_audio(audio_path, model_size="base"):
    """Transcribe using Whisper"""
    if WhisperModel is None:
        return None, "faster-whisper not installed. Run: pip install faster-whisper"
    
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([s.text for s in segments])
        return transcript, None
    except Exception as e:
        return None, str(e)

def generate_summary(text):
    """Generate summary using Ollama"""
    if ollama is None:
        return None, "Ollama not installed. Visit: https://ollama.ai"
    
    try:
        response = ollama.chat(
            model='phi3',
            messages=[
                {
                    'role': 'system',
                    'content': 'Generate a concise bullet-point summary with Action Items, Decisions Made, and Follow-ups.'
                },
                {
                    'role': 'user',
                    'content': text
                }
            ]
        )
        return response['message']['content'], None
    except Exception as e:
        return None, f"Ollama error: {str(e)}. Is Ollama running? Try: ollama serve"

def analyze_document(text, query):
    """Analyze document with Ollama"""
    if ollama is None:
        return None, "Ollama not installed"
    
    try:
        response = ollama.chat(
            model='phi3',
            messages=[
                {
                    'role': 'system',
                    'content': f'Answer based ONLY on this document: {text[:2000]}'
                },
                {
                    'role': 'user',
                    'content': query
                }
            ]
        )
        return response['message']['content'], None
    except Exception as e:
        return None, str(e)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown('<h1 style="text-align: center;">🛡️ EdgeSecure</h1>', unsafe_allow_html=True)
    st.divider()
    
    # Dependency Status
    st.subheader("📊 System Status")
    status_html, deps = get_dependency_status()
    st.markdown(status_html, unsafe_allow_html=True)
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    model_choice = st.selectbox("Whisper Model Size", ["tiny", "base", "small"], help="Larger = more accurate, slower")
    record_duration = st.slider("Recording Duration (sec)", 10, 120, 30)
    
    if st.button("🗑️ Clear Session"):
        st.session_state.clear()
        if os.path.exists(WAV_FILE):
            os.remove(WAV_FILE)
        st.rerun()

# --- 8. MAIN UI ---
st.title("🛡️ EdgeSecure Intelligence Control Center")

# Show warning if critical dependencies missing
if not (deps['faster-whisper'] and deps['ollama']):
    st.warning("""
    ⚠️ **Missing Critical Dependencies**
    
    Run these commands in your terminal:
    ```bash
    pip install faster-whisper ollama sounddevice
    ```
    
    Also, start Ollama in a separate terminal:
    ```bash
    ollama serve
    ```
    
    Then refresh this page.
    """)

t1, t2, t3 = st.tabs(["🎙️ Meeting Scribe", "📄 Vault", "💬 Secure Chat"])

# --- TAB 1: MEETING SCRIBE ---
with t1:
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("🎙️ Recording")
        
        if not st.session_state.recording_active:
            if st.button("🔴 Start Recording", use_container_width=True, key="start_rec"):
                if not sd:
                    st.error("sounddevice not installed")
                else:
                    st.session_state.recording_active = True
                    st.rerun()
        else:
            st.markdown('<p class="recording-status">🔴 RECORDING...</p>', unsafe_allow_html=True)
            st.info(f"Recording for up to {record_duration} seconds...")
            
            if st.button("⏹️ Stop & Transcribe", use_container_width=True, key="stop_rec"):
                with st.spinner("Saving audio..."):
                    audio_file, error = record_audio_simple(record_duration)
                
                if error:
                    st.error(f"Recording error: {error}")
                else:
                    with st.spinner("Transcribing with Whisper..."):
                        transcript, error = transcribe_audio(audio_file, model_choice)
                    
                    if error:
                        st.error(f"Transcription error: {error}")
                    else:
                        st.session_state.transcript = transcript
                        st.session_state.recording_active = False
                        st.rerun()
        
        if st.session_state.transcript:
            st.divider()
            if st.button("📋 Generate Summary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    summary, error = generate_summary(st.session_state.transcript)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state.summary = summary
                    st.rerun()
    
    with c2:
        if st.session_state.transcript:
            st.subheader("📝 Transcript")
            st.info(st.session_state.transcript)
            
            if st.session_state.summary:
                st.divider()
                st.subheader("📊 Summary")
                st.success(st.session_state.summary)

# --- TAB 2: VAULT (PDF Analysis) ---
with t2:
    st.subheader("🔐 Local PDF Vault")
    st.info("Upload PDFs. Analysis happens locally. Never sent to cloud.")
    
    up = st.file_uploader("Upload PDF", type=['pdf'])
    
    if up:
        if st.button("📖 Ingest & Analyze"):
            with st.spinner("Processing PDF..."):
                try:
                    doc = fitz.open(stream=up.read(), filetype="pdf")
                    text = "".join([p.get_text() for p in doc])
                    st.session_state.doc_text = text
                    st.success(f"✅ PDF loaded: {len(text)} characters")
                except Exception as e:
                    st.error(f"PDF error: {e}")
    
    if st.session_state.doc_text:
        st.divider()
        st.subheader("🔍 Query Document")
        query = st.text_input("Ask a question about this document...")
        
        if query and st.button("Search"):
            with st.spinner("Analyzing..."):
                answer, error = analyze_document(st.session_state.doc_text, query)
            
            if error:
                st.error(f"Error: {error}")
            else:
                st.info(answer)

# --- TAB 3: SECURE CHAT ---
with t3:
    st.subheader("💬 Secure Chat (Context-Aware)")
    
    if not st.session_state.transcript and not st.session_state.doc_text:
        st.warning("No context. Capture audio or upload PDF first.")
    else:
        st.info("This chat uses only local data. Nothing leaves your laptop.")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat['q'])
            with st.chat_message("assistant"):
                st.write(chat['a'])
        
        # Chat input
        user_query = st.chat_input("Ask about your meeting or document...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            context = f"Meeting Transcript:\n{st.session_state.transcript[:1000]}\n\nDocument:\n{st.session_state.doc_text[:1000]}"
            
            with st.spinner("Thinking..."):
                answer, error = analyze_document(context, user_query)
            
            if error:
                st.error(f"Error: {error}")
            else:
                with st.chat_message("assistant"):
                    st.write(answer)
                st.session_state.chat_history.append({"q": user_query, "a": answer})

# --- FOOTER ---
st.divider()
st.caption("🛡️ EdgeSecure v1.6 | All processing local | Zero data leakage")