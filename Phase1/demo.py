import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import ollama
import os
import os

# This helps Python find the NVIDIA libraries you just installed
os.environ["PATH"] += os.pathsep + os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python312", "Lib", "site-packages", "nvidia", "cublas", "bin")
os.environ["PATH"] += os.pathsep + os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python312", "Lib", "site-packages", "nvidia", "cudnn", "bin")

# --- CONFIGURATION ---
FS = 44100  # Sample rate
SECONDS = 10  # Duration for testing (change as needed)
FILENAME = "meeting_audio.wav"
MODEL_SIZE = "small"  # 'small' is fast and accurate for 4GB VRAM
# ---------------------

def record_audio():
    print(f"[*] Recording for {SECONDS} seconds...")
    # Record as float32 to be compatible with most AI models
    audio_data = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    write(FILENAME, FS, audio_data)  # Save as .wav
    print(f"[+] Audio saved to {FILENAME}")

def transcribe_audio():
    print("[*] Initializing Whisper model...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    
    print("[*] Transcribing...")
    # ADDED 'initial_prompt': This tells Whisper the context of the conversation
    segments, info = model.transcribe(
        FILENAME, 
        beam_size=5, 
        initial_prompt="A technical meeting discussing Python programming, libraries like sounddevice, scipy, and EdgeSecure software."
    )
    
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
    return full_text.strip()

def summarize_with_phi3(text):
    print("[*] Sending to Phi-3 (Ollama)...")
    
    # IMPROVED PROMPT: Strict instructions to prevent making things up
    system_instruction = "You are a professional meeting assistant. Summarize the transcript accurately. If the transcript contains very little information or is nonsensical, simply state that no meaningful summary can be generated. Do not invent details like names, dates, or budgets."
    
    user_prompt = f"Transcript to summarize:\n{text}"
    
    response = ollama.chat(model='phi3', messages=[
        {'role': 'system', 'content': system_instruction},
        {'role': 'user', 'content': user_prompt},
    ])
    
    return response['message']['content']
if __name__ == "__main__":
    # 1. Capture audio
    record_audio()
    
    # 2. Local Transcription
    transcript = transcribe_audio()
    
    if transcript:
        print(f"\n--- FULL TRANSCRIPT ---\n{transcript}\n")
        
        # 3. Local Summarization
        summary = summarize_with_phi3(transcript)
        print(f"\n--- AI SUMMARY (Phi-3) ---\n{summary}")
    else:
        print("[!] No speech detected.")