import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import math

def format_time(milliseconds):
    """Convert milliseconds to readable time format"""
    seconds = milliseconds / 1000
    minutes = math.floor(seconds / 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02.0f}"

st.title("Audio File Splitter")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    # Load and display original audio
    audio = AudioSegment.from_file(uploaded_file)
    st.audio(uploaded_file)
    
    total_duration = len(audio)
    st.write(f"Total duration: {format_time(total_duration)}")
    
    # Splitting options
    split_method = st.radio(
        "Choose split method:",
        ["Equal parts", "Custom duration", "Manual splits"]
    )
    
    if split_method == "Equal parts":
        num_parts = st.number_input("Number of parts", min_value=2, value=2)
        chunk_duration = total_duration // num_parts
        
    elif split_method == "Custom duration":
        chunk_duration = st.number_input(
            "Duration per chunk (minutes)",
            min_value=1,
            value=5
        ) * 60 * 1000  # Convert to milliseconds
        
    elif split_method == "Manual splits":
        split_points = st.text_input(
            "Enter split points in minutes (comma-separated)",
            "5,10,15"
        )
        split_points = [float(x.strip()) * 60 * 1000 for x in split_points.split(",")]
    
    if st.button("Split Audio"):
        chunks = []
        
        if split_method in ["Equal parts", "Custom duration"]:
            for i in range(0, total_duration, int(chunk_duration)):
                chunk = audio[i:i + chunk_duration]
                chunks.append(chunk)
        else:  # Manual splits
            last_point = 0
            for point in split_points:
                chunk = audio[last_point:point]
                chunks.append(chunk)
                last_point = point
            # Add remaining audio
            if last_point < total_duration:
                chunks.append(audio[last_point:])
        
        # Display and allow download of chunks
        for i, chunk in enumerate(chunks):
            st.write(f"Chunk {i+1} - Duration: {format_time(len(chunk))}")
            
            # Convert chunk to bytes for playback and download
            chunk_file = BytesIO()
            chunk.export(chunk_file, format="wav")
            chunk_file.seek(0)
            
            # Play chunk
            st.audio(chunk_file)
            
            # Download button
            st.download_button(
                f"Download Chunk {i+1}",
                chunk_file,
                file_name=f"chunk_{i+1}.wav",
                mime="audio/wav"
            )
