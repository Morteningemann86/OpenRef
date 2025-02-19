from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pydub import AudioSegment
import time
import streamlit as st

import threading
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx


MAX_CHUNK_SECONDS = 30

class APIQueue:
    def __init__(self):
        self.last_request_time = 0
        self.min_delay = 0.5  # 500ms between requests
        
    def wait_if_needed(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()
        
    def batch_process(self, items, batch_size=3):
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            self.wait_if_needed()
            yield batch

def split_audio(audio_file, chunk_duration=MAX_CHUNK_SECONDS*1000, use_diarization=False):
    """Split audio file into optimized chunks"""
    audio = AudioSegment.from_file(audio_file)
    chunks = []
    
    # Optimize chunk duration for better performance
    if use_diarization:
        overlap_duration = 1000  # 1 second overlap
        chunk_duration = min(chunk_duration, 20000)  # 20 seconds for better performance
    
    for i in range(0, len(audio), chunk_duration):
        chunk = audio[i:i + chunk_duration]
        if use_diarization and i > 0:
            chunk = audio[i-overlap_duration:i + chunk_duration]
        
        chunk_file = BytesIO()
        chunk.export(chunk_file, format="wav")
        chunk_file.seek(0)
        chunk_file.name = "chunk.wav"
        chunks.append(chunk_file)
    
    return chunks

def process_chunk(chunk, transcription_function, use_diarization, queue):
    queue.wait_if_needed()
    try:
        return transcription_function(chunk, use_diarization)
    except Exception as e:
        time.sleep(2)  # Reduced cooldown
        queue.wait_if_needed()
        return transcription_function(chunk, use_diarization)

def process_large_audio(audio_file, transcription_function, use_diarization=False):
    """Process large audio files with optimized queue management and a progress bar"""
    chunks = split_audio(audio_file, use_diarization=use_diarization)
    total_chunks = len(chunks)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    full_transcription = ""
    queue = APIQueue()
    
    # Process chunks in batches without threading
    processed_chunks = 0
    for batch in queue.batch_process(chunks, batch_size=3):
        status_text.write(f"Processing audio... ({processed_chunks + 1}/{total_chunks})")
        
        for chunk in batch:
            queue.wait_if_needed()
            try:
                chunk_transcription = transcription_function(chunk, use_diarization)
                full_transcription += chunk_transcription + "\n"
            except Exception as e:
                time.sleep(2)
                queue.wait_if_needed()
                chunk_transcription = transcription_function(chunk, use_diarization)
                full_transcription += chunk_transcription + "\n"
        
        processed_chunks += len(batch)
        progress = processed_chunks / total_chunks
        progress_bar.progress(progress)
    
    status_text.write("✅ Transcription complete!")
    time.sleep(1)
    progress_bar.empty()
    
    return full_transcription.strip()

