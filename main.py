import streamlit as st
from groq import Groq
import json
import os
from io import BytesIO
from md2pdf.core import md2pdf
from dotenv import load_dotenv
from download import download_video_audio, delete_download, MAX_FILE_SIZE, FILE_TOO_LARGE_MESSAGE
from audio_recorder_streamlit import audio_recorder
from diarization import transcribe_audio_with_speakers
from audio_processing import process_large_audio
from prompt_templates import PROMPT_TEMPLATES


import concurrent.futures  # Import the concurrent futures module
import threading          # Import threading module
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx

# Save the original submit method of ThreadPoolExecutor.
_original_submit = concurrent.futures.ThreadPoolExecutor.submit

def _patched_submit(self, fn, *args, **kwargs):
    # Get the current ScriptRunContext from the main thread.
    ctx = get_script_run_ctx()
    
    # Define a wrapper function that attaches the ScriptRunContext to the worker thread.
    def wrapped_fn(*args, **kwargs):
        add_script_run_ctx(threading.current_thread(), ctx)
        return fn(*args, **kwargs)
    
    # Call the original submit with the wrapped function.
    return _original_submit(self, wrapped_fn, *args, **kwargs)

# Replace the original submit method with our patched version.
concurrent.futures.ThreadPoolExecutor.submit = _patched_submit


# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN", None)

audio_file_path = None

# Initialize session state for API key and Groq client if not already set.
if 'api_key' not in st.session_state:
    st.session_state.api_key = GROQ_API_KEY

if 'groq' not in st.session_state:
    if GROQ_API_KEY:
        st.session_state.groq = Groq()

# Configure the Streamlit page.
st.set_page_config(
    page_title="OpenRef",
    page_icon="👐",
)

# ------------------------------------------------------------------------------
# Function to check Groq API connection.
# ------------------------------------------------------------------------------
def check_groq_api():
    try:
        # Quick test call to Groq API
        st.session_state.groq.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception:
        return False

if 'groq' in st.session_state:
    api_status = check_groq_api()
    if not api_status:
        st.error("⚠️ Groq API is currently unavailable. Please try again later.")
    else:
        st.success("✅ Groq API is connected and ready")

# ------------------------------------------------------------------------------
# GenerationStatistics class: Tracks timing and token usage statistics.
# ------------------------------------------------------------------------------
class GenerationStatistics:
    def __init__(self, input_time=0, output_time=0, input_tokens=0, output_tokens=0, total_time=0, model_name="llama3-8b-8192"):
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time  # Sum of queue, prompt (input), and completion (output) times
        self.model_name = model_name

    def get_input_speed(self):
        if self.input_time != 0:
            return self.input_tokens / self.input_time
        else:
            return 0

    def get_output_speed(self):
        if self.output_time != 0:
            return self.output_tokens / self.output_time
        else:
            return 0

    def add(self, other):
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")
        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    def __str__(self):
        total_tokens = self.input_tokens + self.output_tokens
        cost = total_tokens * 0.000011  # Cost per token
        return (f"\n## {self.get_output_speed():.2f} T/s ⚡\nRound trip time: {self.total_time:.2f}s  Model: {self.model_name}\n"
                f"Total cost: ${cost:.6f}\n\n"
                f"| Metric          | Input          | Output          | Total          |\n"
                f"|-----------------|----------------|-----------------|----------------|\n"
                f"| Speed (T/s)     | {self.get_input_speed():.2f}            | {self.get_output_speed():.2f}            | {(total_tokens) / self.total_time if self.total_time != 0 else 0:.2f}            |\n"
                f"| Tokens          | {self.input_tokens}            | {self.output_tokens}            | {total_tokens}            |\n"
                f"| Cost ($)        | {self.input_tokens * 0.000011:.6f}            | {self.output_tokens * 0.000011:.6f}            | {cost:.6f}            |\n"
                f"| Inference Time (s) | {self.input_time:.2f}            | {self.output_time:.2f}            | {self.total_time:.2f}            |")

# ------------------------------------------------------------------------------
# NoteSection class: Manages note sections and updates content dynamically.
# ------------------------------------------------------------------------------
class NoteSection:
    def __init__(self, structure, transcript):
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}

        st.markdown("## Raw transcript:")
        st.markdown(transcript)
        st.markdown("---")

    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    def update_content(self, title, new_content):
        try:
            self.contents[title] += new_content
            self.display_content(title)
        except TypeError as e:
            pass

    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(f"## {title}\n{self.contents[title]}")

    def return_existing_contents(self, level=1) -> str:
        existing_content = ""
        for title, content in self.structure.items():
            if self.contents[title].strip():
                existing_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                existing_content += self.get_markdown_content(content, level + 1)
        return existing_content

    def display_structure(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
        for title, content in structure.items():
            if self.contents[title].strip():
                st.markdown(f"{'#' * level} {title}")
                self.placeholders[title].markdown(self.contents[title])
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    def display_toc(self, structure, columns, level=1, col_index=0):
        for title, content in structure.items():
            with columns[col_index % len(columns)]:
                st.markdown(f"{' ' * (level-1) * 2}- {title}")
            col_index += 1
            if isinstance(content, dict):
                col_index = self.display_toc(content, columns, level + 1, col_index)
        return col_index

    def get_markdown_content(self, structure=None, level=1):
        if structure is None:
            structure = self.structure
        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}.\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content

# ------------------------------------------------------------------------------
# Utility functions to create downloadable Markdown and PDF files.
# ------------------------------------------------------------------------------
def create_markdown_file(content: str) -> BytesIO:
    markdown_file = BytesIO()
    markdown_file.write(content.encode('utf-8'))
    markdown_file.seek(0)
    return markdown_file

def create_pdf_file(content: str):
    pdf_buffer = BytesIO()
    md2pdf(pdf_buffer, md_content=content)
    pdf_buffer.seek(0)
    return pdf_buffer

# ------------------------------------------------------------------------------
# Function to transcribe audio using Groq's Whisper model.
# ------------------------------------------------------------------------------
def transcribe_audio(audio_file, use_diarization=False):
    try:
        if use_diarization:
            transcript, segments = transcribe_audio_with_speakers(audio_file)
            prompt = "This audio contains multiple speakers"
        else:
            prompt = ""
        # Define the transcription function for each audio chunk.
        def transcription_function(chunk, _):
            if not hasattr(chunk, "name") or not chunk.name:
                chunk.name = "audio_chunk.wav"  # Ensure valid filename
            return st.session_state.groq.audio.transcriptions.create(
                file=chunk,
                model="whisper-large-v3",
                prompt=prompt,
                response_format="json",
                language="en",
                temperature=0.0
            ).text

        full_transcription = process_large_audio(
            audio_file=audio_file,
            transcription_function=transcription_function,
            use_diarization=use_diarization
        )
        return full_transcription

    except Exception as e:
        error_text = str(e)
        if "520" in error_text or "502" in error_text:
            st.error("Groq API is currently experiencing high load. Please wait a few minutes and try again.")
        elif "DOCTYPE html" in error_text:
            st.error("Connection issue detected. Please check your internet connection and try again.")
        else:
            st.error("Error during transcription. Please try again or use a shorter audio clip.")
        return None

# ------------------------------------------------------------------------------
# Function to generate the notes structure using a Llama model.
# ------------------------------------------------------------------------------
def generate_notes_structure(transcript: str, model: str = "llama3-70b-8192"):
    template = PROMPT_TEMPLATES[selected_template]
    completion = st.session_state.groq.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": template["system"] + " Always return valid JSON with string values only, no placeholders."
            },
            {
                "role": "user",
                "content": f"### Transcript {transcript}\n\n### Example\n\n{template['shot_example']}### Instructions\n\nCreate a structure for comprehensive notes on the above transcribed audio. Use only text content, no placeholders."
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

# ------------------------------------------------------------------------------
# Function to generate content for a specific note section.
# ------------------------------------------------------------------------------
def generate_section(transcript: str, existing_notes: str, section: str, model: str = "llama3-8b-8192"):
    stream = st.session_state.groq.chat.completions.create(
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
            statistics_to_return = GenerationStatistics(
                input_time=usage.prompt_time,
                output_time=usage.completion_time,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_time=usage.total_time,
                model_name=model
            )
            yield statistics_to_return

# ------------------------------------------------------------------------------
# Main Streamlit Application Logic
# ------------------------------------------------------------------------------
if 'button_disabled' not in st.session_state:
    st.session_state.button_disabled = False

if 'button_text' not in st.session_state:
    st.session_state.button_text = "Generate Notes"

if 'statistics_text' not in st.session_state:
    st.session_state.statistics_text = ""

st.write("""
# 👐OpenRef: Create structured notes from audio
""")

def disable():
    st.session_state.button_disabled = True

def enable():
    st.session_state.button_disabled = False

def empty_st():
    st.empty()

# ------------------------------------------------------------------------------
# SIDEBAR CUSTOMIZATION SECTION
# ------------------------------------------------------------------------------
try:
    with st.sidebar:
        st.write(f"# 👐 OpenRef \n## Generate notes from audio in seconds using Groq, Whisper, and Llama3")
        st.markdown(f"[Github Repository](https://github.com/Morteningemann86/OpenRef)\n\nAs with all generative AI, content may include inaccurate or placeholder information. OpenRef is an MVP and all feedback is welcome! It is built on top of [ScribeWizard](https://github.com/bklieger/scribewizard)")
        st.write(f"---")
        st.write("# Summary Templates")
        selected_template = st.selectbox(
            "Choose note-taking style:",
            options=list(PROMPT_TEMPLATES.keys())
        )
        st.write(f"---")
        st.write("# Customization Settings\n🧪 These settings are experimental.\n")
        st.write(f"By default, OpenRef uses Llama3-70b for generating the notes outline and Llama3-8b for the content. This balances quality with speed and rate limit usage. You can customize these selections below.")
        outline_model_options = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        outline_selected_model = st.selectbox("Outline generation:", outline_model_options)
        content_model_options = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"]
        content_selected_model = st.selectbox("Content generation:", content_model_options)
        st.info("Important: Different models have different token and rate limits which may cause runtime errors.")

    if st.button('End Generation and Download Notes'):
        if "notes" in st.session_state:
            # Create and offer download for Markdown file
            markdown_file = create_markdown_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download Text',
                data=markdown_file,
                file_name='generated_notes.txt',
                mime='text/plain'
            )
            # Create and offer download for PDF file
            pdf_file = create_pdf_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download PDF',
                data=pdf_file,
                file_name='generated_notes.pdf',
                mime='application/pdf'
            )
            st.session_state.button_disabled = False
        else:
            raise ValueError("Please generate content first before downloading the notes.")

    audio_file = None
    youtube_link = None
    groq_input_key = None

    # ------------------------------------------------------------------------------
    # Input Method Selection: Upload or Record Audio
    # ------------------------------------------------------------------------------
    input_method = st.radio(
        "Choose input method:",
        ["Upload audio file", "Record audio"]
    )

    if input_method == "Upload audio file":
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
        if audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            audio_file.seek(0)
    elif input_method == "Record audio":
        st.write("Click below to record audio")
        audio_bytes = audio_recorder(
            pause_threshold=120.0
        )
        if audio_bytes:
            st.audio(audio_bytes)
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "recording.wav"

    # Toggle to enable/disable speaker diarization
    use_diarization = st.toggle('Enable Speaker Diarization', value=False)

    with st.form("groqform"):
        if not GROQ_API_KEY:
            groq_input_key = st.text_input("Enter your Groq API Key (gsk_yA...):", "", type="password")
        submitted = st.form_submit_button(st.session_state.button_text, on_click=disable, disabled=st.session_state.button_disabled)

        # Status display elements
        status_text = st.empty()
        def display_status(text):
            status_text.write(text)
        def clear_status():
            status_text.empty()

        download_status_text = st.empty()
        def display_download_status(text: str):
            download_status_text.write(text)
        def clear_download_status():
            download_status_text.empty()

        # Placeholder for statistics display
        placeholder = st.empty()
        def display_statistics():
            with placeholder.container():
                if st.session_state.statistics_text:
                    if "Transcribing audio in background" not in st.session_state.statistics_text:
                        st.markdown(st.session_state.statistics_text + "\n\n---\n")
                    else:
                        st.markdown(st.session_state.statistics_text)
                else:
                    placeholder.empty()

        if submitted:
            # Clear previous session data
            if 'notes' in st.session_state:
                del st.session_state.notes
            st.session_state.statistics_text = ""
            placeholder.empty()

            if input_method == "Upload audio file" and audio_file is None:
                st.error("Please upload an audio file")
            elif input_method == "Record audio" and 'audio_recorder' not in st.session_state:
                st.error("Please record some audio first")
            else:
                st.session_state.button_disabled = True

            audio_file_path = None

            if not GROQ_API_KEY:
                st.session_state.groq = Groq(api_key=groq_input_key)

            display_status("Transcribing audio in background....")
            transcription_text = transcribe_audio(audio_file, use_diarization=use_diarization)

            display_statistics()

            display_status("Generating notes structure....")
            large_model_generation_statistics, notes_structure = generate_notes_structure(transcription_text, model=str(outline_selected_model))
            # print("Structure: ", notes_structure)

            display_status("Generating notes ...")
            total_generation_statistics = GenerationStatistics(model_name=str(content_selected_model))
            clear_status()

            try:
                notes_structure_json = json.loads(notes_structure)
                notes = NoteSection(structure=notes_structure_json, transcript=transcription_text)
                if 'notes' not in st.session_state:
                    st.session_state.notes = notes
                st.session_state.notes.display_structure()

                # Recursive function to stream content for each note section.
                def stream_section_content(sections):
                    for title, content in sections.items():
                        if isinstance(content, str):
                            content_stream = generate_section(
                                transcript=transcription_text,
                                existing_notes=notes.return_existing_contents(),
                                section=(title + ": " + content),
                                model=str(content_selected_model)
                            )
                            for chunk in content_stream:
                                # Check if the chunk is GenerationStatistics data.
                                if isinstance(chunk, GenerationStatistics):
                                    total_generation_statistics.add(chunk)
                                    st.session_state.statistics_text = str(total_generation_statistics)
                                    display_statistics()
                                elif chunk is not None:
                                    st.session_state.notes.update_content(title, chunk)
                        elif isinstance(content, dict):
                            stream_section_content(content)
                stream_section_content(notes_structure_json)
            except json.JSONDecodeError:
                st.error("Failed to decode the notes structure. Please try again.")
            enable()
except Exception as e:
    st.session_state.button_disabled = False
    if hasattr(e, 'status_code') and e.status_code == 413:
        st.error(FILE_TOO_LARGE_MESSAGE)
    else:
        st.error(e)
    if st.button("Clear"):
        st.rerun()
    if audio_file_path is not None:
        delete_download(audio_file_path)