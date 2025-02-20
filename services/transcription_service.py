""" 
Handles external API interactions with Groq API for transcription services.
"""
class TranscriptionService:
    def __init__(self, groq_client):
        self.groq_client = groq_client

    def transcribe_audio(self, audio_file):
        transcription = self.groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="json",
            language="en",
            temperature=0.0
        )
        return transcription.text