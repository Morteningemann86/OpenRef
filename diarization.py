import streamlit as st
from groq import Groq
from pyannote.audio import Pipeline
import os
from pydub import AudioSegment
from io import BytesIO
# Import APIQueue from your audio_processing module to manage rate limits
from audio_processing import APIQueue

# Note: The following variables are assumed to be defined elsewhere in your code:
# - PROMPT_TEMPLATES: A dictionary of prompt templates.
# - selected_template: The key for the selected prompt template.
# - GenerationStatistics: A class that captures generation statistics.
# Ensure these are imported or defined in your project.

def transcribe_audio_with_speakers(audio_file):
    """
    Transcribes audio using Groq's Whisper API with speaker diarization,
    applying rate limiting using APIQueue.

    Args:
        audio_file (BytesIO or str): The audio file to transcribe. If a BytesIO object is
                                     provided, it will be converted to a temporary WAV file.

    Returns:
        formatted_transcript (str): A transcript with speaker labels.
        transcribed_segments (list): A list of dictionaries containing speaker, text,
                                     start, and end times for each segment.
    """
    # Initialize the speaker diarization pipeline from Hugging Face.
    # Replace 'YOUR_HUGGING_FACE_TOKEN' with your actual Hugging Face token.
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=os.environ.get("HF_TOKEN")
    )

    # If audio_file is a BytesIO stream, export it to a temporary WAV file.
    if isinstance(audio_file, BytesIO):
        audio = AudioSegment.from_file(audio_file)
        audio.export("temp_audio.wav", format="wav")
        audio_file_path = "temp_audio.wav"
    else:
        # Otherwise, assume audio_file is a file path.
        audio_file_path = audio_file

    # Perform speaker diarization on the provided audio file.
    diarization = pipeline(audio_file_path)
    
    # Extract segments with speaker labels.
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,  # Start time in seconds
            "end": turn.end,      # End time in seconds
            "speaker": speaker    # Speaker label
        })

    # Initialize the APIQueue to enforce rate limiting for API calls.
    queue = APIQueue()

    # List to hold transcriptions for each segment.
    transcribed_segments = []
    for segment in segments:
        # Calculate start and end times in milliseconds.
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        
        # Extract the audio segment corresponding to the current speaker turn.
        audio_segment = AudioSegment.from_wav(audio_file_path)[start_ms:end_ms]
        
        # Export the segment to a temporary WAV file.
        audio_segment.export("temp_segment.wav", format="wav")
        
        # Wait until the API is available (rate limiting).
        queue.wait_if_needed()
        
        # Transcribe the audio segment using Groq's Whisper API.
        with open("temp_segment.wav", "rb") as segment_file:
            transcription = st.session_state.groq.audio.transcriptions.create(
                file=segment_file,
                model="whisper-large-v3",
                prompt="",
                response_format="json",
                language="en",
                temperature=0.0
            )
        
        # Store the transcription along with speaker and timing info.
        transcribed_segments.append({
            "speaker": segment["speaker"],
            "text": transcription.text,
            "start": segment["start"],
            "end": segment["end"]
        })

    # Clean up temporary files if they exist.
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    if os.path.exists("temp_segment.wav"):
        os.remove("temp_segment.wav")

    # Construct a formatted transcript that includes speaker labels.
    formatted_transcript = ""
    for segment in transcribed_segments:
        formatted_transcript += f"[{segment['speaker']}]: {segment['text']}\n"

    return formatted_transcript, transcribed_segments

def generate_notes_structure(transcript: str, segments: list, model: str = "llama3-70b-8192"):
    """
    Generates a notes structure using Groq's Chat API that takes into account speaker information.

    Args:
        transcript (str): The full transcript of the audio.
        segments (list): List of speaker-segment dictionaries.
        model (str): The model name to use for generating the structure.

    Returns:
        statistics_to_return (GenerationStatistics): Statistics from the API call.
        notes_structure (str): The generated notes structure in JSON format.
    """
    # Ensure that PROMPT_TEMPLATES and selected_template are defined in your project.
    template = PROMPT_TEMPLATES[selected_template]
    
    # Build a context prompt that informs the model about multiple speakers.
    speaker_context = (
        "This transcript includes multiple speakers. Please consider the different "
        "perspectives and contributions from each speaker when creating the structure.\n\n"
    )
    
    # Create the completion call using Groq's Chat API.
    completion = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    template["system"] +
                    " Always return valid JSON with string values only, no placeholders. "
                    "Consider speaker roles and perspectives in the structure."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{speaker_context}### Transcript\n{transcript}\n\n### Example\n\n"
                    f"{template['shot_example']}### Instructions\n\n"
                    "Create a structure for comprehensive notes on the above transcribed audio, "
                    "considering the different speakers and their contributions. Use only text content, no placeholders."
                )
            }
        ],
        temperature=0.2,       # Lower temperature for consistency
        max_tokens=8000,       # Maximum tokens for the response
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    # Retrieve usage statistics from the API call.
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