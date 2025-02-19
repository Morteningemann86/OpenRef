from pydub import AudioSegment
from io import BytesIO
import streamlit as st
import time

# Groq Whisper API limits
CALLS_PER_MINUTE = 20
SECONDS_PER_HOUR = 7200
SECONDS_PER_DAY = 28800
MAX_CHUNK_SECONDS = 60

class APIQueue:
    def __init__(self):
        self.last_request_time = 0
        self.requests_this_minute = 0
        
    def wait_if_needed(self):
        current_time = time.time()
        if current_time - self.last_request_time < 60:  # Still in same minute
            if self.requests_this_minute >= CALLS_PER_MINUTE:
                sleep_time = 60 - (current_time - self.last_request_time)
                st.write(f"Waiting {sleep_time:.1f} seconds for API cooldown...")
                time.sleep(sleep_time)
                self.requests_this_minute = 0
        else:  # New minute started
            self.requests_this_minute = 0
            
        self.last_request_time = current_time
        self.requests_this_minute += 1

def split_audio(audio_file, chunk_duration=MAX_CHUNK_SECONDS*1000):
    """Split audio file into chunks that respect Groq's audio duration limits"""
    audio = AudioSegment.from_file(audio_file)
    chunks = []
    total_duration = len(audio) / 1000
    
    if total_duration > SECONDS_PER_DAY:
        st.warning(f"Audio duration ({total_duration:.1f}s) exceeds daily limit ({SECONDS_PER_DAY}s)")
    
    for i in range(0, len(audio), chunk_duration):
        chunk = audio[i:i + chunk_duration]
        chunk_file = BytesIO()
        chunk.export(chunk_file, format="wav")
        chunk_file.seek(0)
        chunk_file.name = "chunk.wav"
        chunks.append(chunk_file)
    
    return chunks

def process_large_audio(audio_file, transcription_function, use_diarization=False):
    """Process large audio files with intelligent API queue management and a progress bar"""
    chunks = split_audio(audio_file)
    total_chunks = len(chunks)
    
    # Initialize the progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()  # Create an empty placeholder for status updates
    full_transcription = ""
    queue = APIQueue()
    
    for i, chunk in enumerate(chunks):
        status_text.write(f"Processing audio... ({i+1}/{total_chunks})")

        # Wait for API availability
        queue.wait_if_needed()

        try:
            chunk_transcription = transcription_function(chunk, use_diarization)
            full_transcription += chunk_transcription + "\n"
        except Exception as e:
            status_text.write("API error detected, retrying...")
            time.sleep(5)  # Brief cooldown
            queue.wait_if_needed()
            chunk_transcription = transcription_function(chunk, use_diarization)
            full_transcription += chunk_transcription + "\n"

        # Update progress bar
        progress = (i + 1) / total_chunks  # Calculate progress percentage
        progress_bar.progress(progress)

    # Mark completion
    status_text.write("✅ Transcription complete!")
    time.sleep(1)  # Small delay before clearing the progress bar
    progress_bar.empty()  # Remove the progress bar after completion

    return full_transcription
