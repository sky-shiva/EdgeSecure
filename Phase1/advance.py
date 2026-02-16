import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import ollama
import time
from datetime import datetime

# --- SETTINGS ---
FS = 44100 
FILENAME = "meeting_audio.wav"
# Using 'tiny' or 'base' makes CPU transcription lightning fast
MODEL_SIZE = "base" 

def record_until_stopped():
    print("\n" + "="*40)
    print(" EDGESECURE - LOCAL AI RECORDER")
    print("="*40)
    print("[!] RECORDING... Press Ctrl+C to stop.")
    
    audio_list = []
    try:
        def callback(indata, frames, time, status):
            audio_list.append(indata.copy())
            
        with sd.InputStream(samplerate=FS, channels=1, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[+] Recording stopped.")
        audio_data = np.concatenate(audio_list, axis=0)
        write(FILENAME, FS, audio_data)

def transcribe_audio():
    print("[*] Initializing stable CPU engine...")
    # 'cpu' and 'int8' are the secret to speed without a GPU
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    
    print("[*] Transcribing (Local-Only)...")
    segments, _ = model.transcribe(FILENAME, beam_size=5)
    
    text = " ".join([s.text for s in segments]).strip()
    return text

def summarize_and_save(transcript):
    print("[*] Analyzing with Phi-3 (Ollama)...")
    
    prompt = f"Professional Summary for EdgeSecure Project:\n\n{transcript}"
    
    try:
        response = ollama.chat(model='phi3', messages=[
            {'role': 'system', 'content': 'Summarize the meeting accurately in bullet points.'},
            {'role': 'user', 'content': prompt}
        ])
        summary = response['message']['content']
    except Exception as e:
        summary = f"Ollama Error: Ensure Ollama is running in the background. {e}"

    # Save the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_name = f"EdgeSecure_{timestamp}.txt"
    
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(f"EDGESECURE MEETING REPORT\nGenerated: {timestamp}\n")
        f.write("="*30 + "\n\n")
        f.write(f"TRANSCRIPT:\n{transcript}\n\n")
        f.write(f"SUMMARY:\n{summary}\n")
    
    return report_name, summary

if __name__ == "__main__":
    # 1. Capture
    record_until_stopped()
    
    # 2. Transcribe
    full_text = transcribe_audio()
    
    if full_text:
        print(f"\n[TRANSCRIPT]: {full_text}")
        
        # 3. Summarize
        fname, final_summary = summarize_and_save(full_text)
        
        print("\n" + "="*40)
        print(f"REPORT GENERATED: {fname}")
        print("="*40)
        print(final_summary)
    else:
        print("[!] No audio was captured.")