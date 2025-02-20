"""
Handles external API interactions with Groq API.
"""
from typing import Tuple, Generator, Union
from groq import Groq
import json
from models.statistics import GenerationStatistics


class GroqService:
    def __init__(self, api_key: str):
        """Initialize Groq client with API key."""
        self.client = Groq(api_key=api_key)

    def generate_notes_structure(self, transcript: str, model: str) -> Tuple[GenerationStatistics, dict]:
        """
        Generate structure for notes from transcript.
        
        Args:
            transcript (str): The transcribed text
            model (str): The model to use for generation
            
        Returns:
            Tuple[GenerationStatistics, dict]: Statistics and generated structure
        """
        shot_example = """
        "Introduction": "Introduction to the AMA session, including the topic of Groq scaling architecture and the panelists",
        "Panelist Introductions": "Brief introductions from Igor, Andrew, and Omar, covering their backgrounds and roles at Groq",
        "Groq Scaling Architecture Overview": "High-level overview of Groq's scaling architecture, covering hardware, software, and cloud components"
        }"""

        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Write in JSON format:\n\n{\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\"}"
                },
                {
                    "role": "user",
                    "content": f"### Transcript {transcript}\n\n### Example\n\n{shot_example}### Instructions\n\nCreate a structure for comprehensive notes on the above transcribed audio. Section titles and content descriptions must be comprehensive. Quality over quantity."
                }
            ],
            temperature=0.3,
            max_tokens=8000,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )

        usage = completion.usage
        statistics = GenerationStatistics(
            input_time=usage.prompt_time,
            output_time=usage.completion_time,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_time=usage.total_time,
            model_name=model
        )

        notes_structure = json.loads(completion.choices[0].message.content)
        return statistics, notes_structure

    def generate_section(
        self,
        transcript: str,
        existing_notes: str,
        section: str,
        model: str
    ) -> Generator[Union[str, GenerationStatistics], None, None]:
        """
        Generate content for a specific section.
        
        Args:
            transcript (str): The transcribed text
            existing_notes (str): Currently generated notes
            section (str): The section to generate content for
            model (str): The model to use for generation
            
        Yields:
            Union[str, GenerationStatistics]: Either content chunks or generation statistics
        """
        stream = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert writer. Generate a comprehensive note for the section provided based factually on the transcript provided. Do *not* repeat any content from previous sections."
                },
                {
                    "role": "user",
                    "content": f"### Transcript\n\n{transcript}\n\n### Existing Notes\n\n{existing_notes}\n\n### Instructions\n\nGenerate comprehensive notes for this section only based on the transcript: \n\n{section}"
                }
            ],
            temperature=0.3,
            max_tokens=8000,
            top_p=1,
            stream=True,
            stop=None,
        )

        for chunk in stream:
            tokens = chunk.choices[0].delta.content
            if tokens:
                yield tokens

            if x_groq := chunk.x_groq:
                if not x_groq.usage:
                    continue
                    
                usage = x_groq.usage
                statistics = GenerationStatistics(
                    input_time=usage.prompt_time,
                    output_time=usage.completion_time,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_time=usage.total_time,
                    model_name=model
                )
                yield statistics

    def transcribe_audio(self, audio_file) -> str:
        """
        Transcribe audio using Groq's Whisper API.
        
        Args:
            audio_file: The audio file to transcribe
            
        Returns:
            str: The transcribed text
        """
        transcription = self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            prompt="",
            response_format="json",
            language="en",
            temperature=0.0 
        )

        return transcription.text