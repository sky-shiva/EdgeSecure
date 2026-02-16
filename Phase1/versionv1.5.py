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
st.set_page_config(page_title="EdgeSecure AI DEBUG", page_icon="🛡️", layout="wide")
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
    .debug-box { background-color: #1a1a2e; border: 1px solid #00d4ff; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.85rem; margin: 10px 0; }
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
if 'debug_log' not in st.session_state:
    st.session_state.debug_log = []
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 16000

def debug_log(message):
    """Add message to debug log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    st.session_state.debug_log.append(full_msg)
    print(full_msg)  # Also print to console

# --- 4. DEPENDENCY CHECK ---
def check_dependencies():
    """Check which dependencies are installed"""
    deps = {
        'sounddevice': sd is not None,
        'faster-whisper': WhisperModel is not None,
        'ollama': ollama is not None,
        'PyMuPDF': True,
        'streamlit': True,
    }
    return deps

# --- 5. AUDIO RECORDING WITH DEBUG ---
def record_audio_simple(duration=60):
    """Simple audio recording using sounddevice (cross-platform)"""
    debug_log(f"Starting audio recording for {duration} seconds...")
    
    if sd is None:
        debug_log("ERROR: sounddevice not installed")
        return None, "sounddevice not installed. Run: pip install sounddevice"
    
    try:
        sample_rate = 16000
        st.session_state.sample_rate = sample_rate
        debug_log(f"Sample rate: {sample_rate} Hz")
        
        # List audio devices
        try:
            devices = sd.query_devices()
            debug_log(f"Found {len(devices)} audio devices")
            default_input = sd.default.device[0]
            debug_log(f"Default input device: {default_input}")
        except Exception as e:
            debug_log(f"Warning: Could not list devices: {e}")
        
        # Record audio
        debug_log("Recording audio...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()  # Wait for recording to finish
        debug_log(f"Recording complete. Data shape: {audio_data.shape}")
        
        # Save to WAV
        debug_log(f"Saving to: {WAV_FILE}")
        with wave.open(WAV_FILE, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        # Verify file exists
        if os.path.exists(WAV_FILE):
            file_size = os.path.getsize(WAV_FILE)
            debug_log(f"✅ WAV file created. Size: {file_size} bytes")
            return WAV_FILE, None
        else:
            debug_log("❌ WAV file was not created!")
            return None, "WAV file creation failed"
            
    except Exception as e:
        debug_log(f"❌ Recording error: {type(e).__name__}: {str(e)}")
        return None, f"Recording failed: {str(e)}"

# --- 6. TRANSCRIPTION WITH DEBUG ---
def transcribe_audio(audio_path, model_size="base"):
    """Transcribe using Whisper with detailed logging"""
    debug_log(f"Starting transcription with model: {model_size}")
    
    if WhisperModel is None:
        debug_log("ERROR: faster-whisper not installed")
        return None, "faster-whisper not installed. Run: pip install faster-whisper"
    
    if not os.path.exists(audio_path):
        debug_log(f"❌ Audio file not found: {audio_path}")
        return None, f"Audio file not found: {audio_path}"
    
    try:
        file_size = os.path.getsize(audio_path)
        debug_log(f"Audio file size: {file_size} bytes")
        
        if file_size < 1000:
            debug_log("⚠️  WARNING: Audio file is very small (might be silent)")
        
        debug_log(f"Loading Whisper model: {model_size}...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        debug_log("✅ Model loaded")
        
        debug_log("Running transcription...")
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        # Collect segments
        transcript_list = []
        for i, segment in enumerate(segments):
            debug_log(f"  Segment {i}: {segment.text[:50]}...")
            transcript_list.append(segment.text)
        
        transcript = " ".join(transcript_list)
        debug_log(f"✅ Transcription complete. Total text length: {len(transcript)} chars")
        
        if len(transcript.strip()) == 0:
            debug_log("⚠️  WARNING: Transcript is empty (audio might be silent)")
        
        return transcript, None
        
    except Exception as e:
        debug_log(f"❌ Transcription error: {type(e).__name__}: {str(e)}")
        import traceback
        debug_log(f"Traceback: {traceback.format_exc()}")
        return None, f"Transcription failed: {str(e)}"

# --- 7. SUMMARY GENERATION WITH DEBUG ---
def generate_summary(text):
    """Generate summary using Ollama with debug"""
    debug_log("Generating summary...")
    
    if ollama is None:
        debug_log("ERROR: Ollama not installed")
        return None, "Ollama not installed. Visit: https://ollama.ai"
    
    try:
        debug_log("Connecting to Ollama...")
        response = ollama.chat(
            model='phi3',
            messages=[
                {
                    'role': 'system',
                    'content': 'Generate a concise bullet-point summary with Action Items, Decisions Made, and Follow-ups.'
                },
                {
                    'role': 'user',
                    'content': text[:2000]  # Limit input size
                }
            ]
        )
        debug_log("✅ Summary generated")
        return response['message']['content'], None
    except Exception as e:
        debug_log(f"❌ Summary error: {type(e).__name__}: {str(e)}")
        return None, f"Summary failed: {str(e)}. Make sure Ollama is running: ollama serve"

# --- 8. SIDEBAR ---
with st.sidebar:
    st.markdown('<h1 style="text-align: center;">🛡️ EdgeSecure DEBUG</h1>', unsafe_allow_html=True)
    st.divider()
    
    # Dependency Status
    st.subheader("📊 System Status")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        st.write(f"{status} {dep}")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    model_choice = st.selectbox("Whisper Model Size", ["tiny", "base", "small"])
    record_duration = st.slider("Recording Duration (sec)", 5, 60, 10)
    
    st.divider()
    st.subheader("🔧 Debug Options")
    
    if st.button("📋 Show Full Debug Log"):
        st.session_state.show_debug = not st.session_state.get('show_debug', False)
    
    if st.button("🗑️ Clear Session"):
        st.session_state.clear()
        if os.path.exists(WAV_FILE):
            os.remove(WAV_FILE)
        st.rerun()

# --- 9. MAIN UI ---
st.title("🛡️ EdgeSecure DEBUG Mode")

t1, t2, t3, t4 = st.tabs(["🎙️ Meeting Scribe", "📄 Vault", "💬 Chat", "🔍 Debug Log"])

# --- TAB 1: MEETING SCRIBE ---
with t1:
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("🎙️ Recording")
        
        if not st.session_state.recording_active:
            if st.button("🔴 Start Recording", use_container_width=True, key="start_rec"):
                if not sd:
                    st.error("sounddevice not installed")
                    debug_log("ERROR: User tried to record but sounddevice not installed")
                else:
                    st.session_state.recording_active = True
                    st.rerun()
        else:
            st.markdown('<p class="recording-status">🔴 RECORDING...</p>', unsafe_allow_html=True)
            st.info(f"Recording for {record_duration} seconds... Click below to stop when done.")
            
            if st.button("⏹️ Stop & Transcribe", use_container_width=True, key="stop_rec"):
                with st.spinner("Saving audio..."):
                    audio_file, error = record_audio_simple(record_duration)
                
                if error:
                    st.error(f"❌ {error}")
                    debug_log(f"Recording failed: {error}")
                else:
                    st.success("✅ Audio saved")
                    
                    with st.spinner("Transcribing with Whisper..."):
                        transcript, error = transcribe_audio(audio_file, model_choice)
                    
                    if error:
                        st.error(f"❌ {error}")
                        debug_log(f"Transcription failed: {error}")
                    else:
                        st.success("✅ Transcription complete")
                        st.session_state.transcript = transcript
                        st.session_state.recording_active = False
                        st.rerun()
        
        if st.session_state.transcript:
            st.divider()
            if st.button("📋 Generate Summary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    summary, error = generate_summary(st.session_state.transcript)
                
                if error:
                    st.error(f"❌ {error}")
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
        else:
            st.warning("No transcript yet. Record something first.")

# --- TAB 2: VAULT ---
with t2:
    st.subheader("📄 Vault")
    st.write("PDF upload feature here (same as v1.6)")

# --- TAB 3: CHAT ---
with t3:
    st.subheader("💬 Chat")
    st.write("Chat feature here (same as v1.6)")

# --- TAB 4: DEBUG LOG ---
with t4:
    st.subheader("🔍 Debug Log")
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    if st.button("📋 Copy All Logs"):
        log_text = "\n".join(st.session_state.debug_log)
        st.code(log_text, language="text")
    
    st.markdown("### Recent Logs (Last 20)")
    log_text = "\n".join(st.session_state.debug_log[-20:]) if st.session_state.debug_log else "No logs yet"
    
    st.markdown(f'<div class="debug-box">{log_text}</div>', unsafe_allow_html=True)

st.divider()
st.caption("🛡️ EdgeSecure DEBUG v1.6.1 | All processing local | Zero data leakage")