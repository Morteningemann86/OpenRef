"""
Houses helper functions and constants for file handling.
"""
from io import BytesIO
from md2pdf.core import md2pdf

def create_markdown_file(content: str) -> BytesIO:
    markdown_file = BytesIO()
    markdown_file.write(content.encode('utf-8'))
    markdown_file.seek(0)
    return markdown_file

def create_pdf_file(content: str) -> BytesIO:
    pdf_buffer = BytesIO()
    md2pdf(pdf_buffer, md_content=content)
    pdf_buffer.seek(0)
    return pdf_buffer