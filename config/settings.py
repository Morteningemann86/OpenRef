import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", None)
DEFAULT_OUTLINE_MODEL = "llama3-70b-8192"
DEFAULT_CONTENT_MODEL = "llama3-8b-8192"
MODEL_OPTIONS = {
    "outline": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    "content": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"]
}