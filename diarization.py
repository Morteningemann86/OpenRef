import streamlit as st
from groq import Groq
from pyannote.audio import Pipeline
import torch
import numpy as np
import wave
from pydub import AudioSegment
import json
import os
from io import BytesIO

def transcribe_audio_with_speakers(audio_file):
    """
    Transcribes audio using Groq's Whisper API with speaker diarization.
    """
    # Initialize pyannote pipeline for speaker diarization
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="YOUR_HUGGING_FACE_TOKEN"  # You'll need to get this from Hugging Face
    )

    # Convert audio to format compatible with pyannote
    if isinstance(audio_file, BytesIO):
        audio = AudioSegment.from_file(audio_file)
        audio.export("temp_audio.wav", format="wav")
        audio_file_path = "temp_audio.wav"
    else:
        audio_file_path = audio_file

    # Perform speaker diarization
    diarization = pipeline(audio_file_path)
    
    # Create segments with speaker information
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # Transcribe each segment
    transcribed_segments = []
    for segment in segments:
        # Extract audio segment
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        audio_segment = AudioSegment.from_wav(audio_file_path)[start_ms:end_ms]
        
        # Save segment temporarily
        audio_segment.export("temp_segment.wav", format="wav")
        
        # Transcribe segment
        with open("temp_segment.wav", "rb") as segment_file:
            transcription = st.session_state.groq.audio.transcriptions.create(
                file=segment_file,
                model="whisper-large-v3",
                prompt="",
                response_format="json",
                language="en",
                temperature=0.0
            )
        
        transcribed_segments.append({
            "speaker": segment["speaker"],
            "text": transcription.text,
            "start": segment["start"],
            "end": segment["end"]
        })

    # Clean up temporary files
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    if os.path.exists("temp_segment.wav"):
        os.remove("temp_segment.wav")

    # Format the final transcript with speaker labels
    formatted_transcript = ""
    for segment in transcribed_segments:
        formatted_transcript += f"[{segment['speaker']}]: {segment['text']}\n"

    return formatted_transcript, transcribed_segments

def generate_notes_structure(transcript: str, segments: list, model: str = "llama3-70b-8192"):
    """
    Modified to include speaker information in the structure generation
    """
    template = PROMPT_TEMPLATES[selected_template]
    
    # Create a more detailed prompt that includes speaker information
    speaker_context = "This transcript includes multiple speakers. Please consider the different perspectives and contributions from each speaker when creating the structure.\n\n"
    
    completion = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": template["system"] + " Always return valid JSON with string values only, no placeholders. Consider speaker roles and perspectives in the structure."
            },
            {
                "role": "user",
                "content": f"{speaker_context}### Transcript\n{transcript}\n\n### Example\n\n{template['shot_example']}### Instructions\n\nCreate a structure for comprehensive notes on the above transcribed audio, considering the different speakers and their contributions. Use only text content, no placeholders."
            }
        ],
        temperature=0.2,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics_to_return = GenerationStatistics(
        input_time=usage.prompt_time,
        output_time=usage.completion_time,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_time=usage.total_time,
        model_name=model
    )

    return statistics_to_return, completion.choices[0].message.content