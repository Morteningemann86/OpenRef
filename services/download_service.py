import os
import tempfile
from typing import Optional, Callable
from pytube import YouTube
from urllib.error import URLError

# Constants
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
FILE_TOO_LARGE_MESSAGE = (
    "File size exceeds the maximum limit of 25MB. "
    "This limitation exists because the Whisper API has a maximum file size limit. "
    "In the future, ScribeWizard will automatically split larger files."
)

def download_video_audio(
    youtube_url: str,
    status_callback: Optional[Callable[[str], None]] = None
) -> Optional[str]:
    """
    Download audio from a YouTube video.
    
    Args:
        youtube_url (str): The URL of the YouTube video
        status_callback (Optional[Callable[[str], None]]): Optional callback for status updates
        
    Returns:
        Optional[str]: Path to the downloaded audio file or None if download fails
    """
    try:
        # Create YouTube object
        yt = YouTube(youtube_url)
        
        # Get audio stream
        if status_callback:
            status_callback("Finding best audio stream...")
        
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            if status_callback:
                status_callback("No audio stream found.")
            return None
            
        # Create temporary directory for download
        temp_dir = tempfile.mkdtemp()
        
        # Download audio
        if status_callback:
            status_callback(f"Downloading audio from: {yt.title}")
            
        download_path = audio_stream.download(temp_dir)
        
        # Rename file to ensure .mp3 extension
        base, _ = os.path.splitext(download_path)
        new_path = base + '.mp3'
        os.rename(download_path, new_path)
        
        return new_path
        
    except URLError:
        if status_callback:
            status_callback("Failed to connect to YouTube. Please check your internet connection.")
        return None
        
    except Exception as e:
        if status_callback:
            status_callback(f"Error downloading video: {str(e)}")
        return None

def delete_download(file_path: str) -> None:
    """
    Delete a downloaded file and its parent directory if empty.
    
    Args:
        file_path (str): Path to the file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Try to remove parent directory if empty
        parent_dir = os.path.dirname(file_path)
        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
            os.rmdir(parent_dir)
            
    except Exception as e:
        print(f"Error cleaning up downloaded file: {str(e)}")