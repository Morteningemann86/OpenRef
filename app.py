import streamlit as st
from typing import Optional, Tuple
from io import BytesIO
import os

from config.settings import (
    GROQ_API_KEY,
    MODEL_OPTIONS,
    DEFAULT_OUTLINE_MODEL,
    DEFAULT_CONTENT_MODEL
)
from services.groq_service import GroqService
from services.transcription_service import TranscriptionService
from services.download_service import download_video_audio, delete_download, MAX_FILE_SIZE, FILE_TOO_LARGE_MESSAGE
from ui.components import NoteSection
from utils.file_handlers import create_markdown_file, create_pdf_file
from models.statistics import GenerationStatistics

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = GROQ_API_KEY
    
    if 'groq' not in st.session_state:
        if GROQ_API_KEY:
            st.session_state.groq = GroqService(GROQ_API_KEY)
    
    if 'button_disabled' not in st.session_state:
        st.session_state.button_disabled = False
    
    if 'button_text' not in st.session_state:
        st.session_state.button_text = "Generate Notes"
    
    if 'statistics_text' not in st.session_state:
        st.session_state.statistics_text = ""

def setup_sidebar() -> Tuple[str, str]:
    """Setup sidebar UI and return selected models."""
    with st.sidebar:
        st.write("# 🧙‍♂️ ScribeWizard")
        st.write("## Generate notes from audio in seconds using Groq, Whisper, and Llama3")
        st.markdown("[Github Repository](https://github.com/bklieger/scribewizard)\n\n"
                   "As with all generative AI, content may include inaccurate or placeholder information. "
                   "ScribeWizard is in beta and all feedback is welcome!")
        
        st.write("---")
        
        st.write("# Customization Settings\n🧪 These settings are experimental.\n")
        st.write("By default, ScribeWizard uses Llama3-70b for generating the notes outline and "
                "Llama3-8b for the content. This balances quality with speed and rate limit usage.")
        
        outline_selected_model = st.selectbox(
            "Outline generation:",
            MODEL_OPTIONS["outline"],
            index=MODEL_OPTIONS["outline"].index(DEFAULT_OUTLINE_MODEL)
        )
        
        content_selected_model = st.selectbox(
            "Content generation:",
            MODEL_OPTIONS["content"],
            index=MODEL_OPTIONS["content"].index(DEFAULT_CONTENT_MODEL)
        )
        
        st.info("Important: Different models have different token and rate limits which may cause runtime errors.")
        
    return outline_selected_model, content_selected_model

def handle_downloads():
    """Handle note download functionality."""
    if st.button('End Generation and Download Notes'):
        if "notes" in st.session_state:
            # Create markdown file
            markdown_file = create_markdown_file(st.session_state.notes.get_markdown_content())
            st.download_button(
                label='Download Text',
                data=markdown_file,
                file_name='generated_notes.txt',
                mime='text/plain'
            )

            # Create pdf file
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

def process_input(input_method: str, status_placeholder) -> Optional[BytesIO]:
    """Process user input (audio file or YouTube link) and return audio file."""
    audio_file = None
    audio_file_path = None
    
    try:
        if input_method == "Upload audio file":
            audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
            if not audio_file:
                st.error("Please upload an audio file")
                return None
                
        else:  # YouTube link
            youtube_link = st.text_input("Enter YouTube link:", "")
            if not youtube_link:
                st.error("Please enter a YouTube link")
                return None
                
            status_placeholder.write("Downloading audio from YouTube link ....")
            audio_file_path = download_video_audio(youtube_link, lambda x: status_placeholder.write(x))
            
            if audio_file_path is None:
                st.error("Failed to download audio from YouTube link. Please try again.")
                return None
                
            # Read the downloaded file
            status_placeholder.write("Processing Youtube audio ....")
            with open(audio_file_path, 'rb') as f:
                file_contents = f.read()
            audio_file = BytesIO(file_contents)
            
            # Check file size
            if os.path.getsize(audio_file_path) > MAX_FILE_SIZE:
                raise ValueError(FILE_TOO_LARGE_MESSAGE)
                
            audio_file.name = os.path.basename(audio_file_path)
            
        return audio_file
        
    finally:
        if audio_file_path:
            delete_download(audio_file_path)
        status_placeholder.empty()

def generate_notes(
    audio_file: BytesIO,
    outline_model: str,
    content_model: str,
    status_placeholder,
    stats_placeholder
) -> None:
    """Generate notes from audio file."""
    transcription_service = TranscriptionService(st.session_state.groq.client)
    
    # Transcribe audio
    status_placeholder.write("Transcribing audio in background....")
    transcription_text = transcription_service.transcribe_audio(audio_file)
    
    # Generate structure
    status_placeholder.write("Generating notes structure....")
    structure_stats, notes_structure = st.session_state.groq.generate_notes_structure(
        transcription_text,
        model=outline_model
    )
    
    # Generate content
    status_placeholder.write("Generating notes...")
    total_stats = GenerationStatistics(model_name=content_model)
    
    # Create and store notes
    notes = NoteSection(structure=notes_structure, transcript=transcription_text)
    st.session_state.notes = notes
    notes.display_structure()
    
    # Generate content for each section
    def stream_section_content(sections):
        for title, content in sections.items():
            if isinstance(content, str):
                content_stream = st.session_state.groq.generate_section(
                    transcript=transcription_text,
                    existing_notes=notes.return_existing_contents(),
                    section=f"{title}: {content}",
                    model=content_model
                )
                
                for chunk in content_stream:
                    if isinstance(chunk, GenerationStatistics):
                        total_stats.add(chunk)
                        st.session_state.statistics_text = str(total_stats)
                        stats_placeholder.markdown(st.session_state.statistics_text + "\n\n---\n")
                    elif chunk:
                        notes.update_content(title, chunk)
            elif isinstance(content, dict):
                stream_section_content(content)
    
    stream_section_content(notes_structure)
    status_placeholder.empty()

def main():
    """Main application function."""
    st.set_page_config(page_title="ScribeWizard", page_icon="🧙‍♂️")
    
    # Initialize session state
    initialize_session_state()
    
    # Page title
    st.write("# ScribeWizard: Create structured notes from audio 🗒️⚡")
    
    # Setup sidebar and get model selections
    outline_model, content_model = setup_sidebar()
    
    # Handle downloads
    handle_downloads()
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload audio file", "YouTube link"])
    
    # Create status placeholders
    status_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Form for input and generation
    with st.form("groqform"):
        # Handle Groq API key if not set
        if not GROQ_API_KEY:
            groq_input_key = st.text_input("Enter your Groq API Key (gsk_yA...):", "", type="password")
            if groq_input_key:
                st.session_state.groq = GroqService(groq_input_key)
        
        # Process input
        audio_file = process_input(input_method, status_placeholder)
        
        # Generate button
        submitted = st.form_submit_button(
            st.session_state.button_text,
            disabled=st.session_state.button_disabled
        )
        
        if submitted and audio_file:
            try:
                st.session_state.button_disabled = True
                generate_notes(
                    audio_file,
                    outline_model,
                    content_model,
                    status_placeholder,
                    stats_placeholder
                )
            except Exception as e:
                st.session_state.button_disabled = False
                if hasattr(e, 'status_code') and e.status_code == 413:
                    st.error(FILE_TOO_LARGE_MESSAGE)
                else:
                    st.error(str(e))
                
                if st.button("Clear"):
                    st.rerun()
            finally:
                st.session_state.button_disabled = False

if __name__ == "__main__":
    main()