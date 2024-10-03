import fitz  # PyMuPDF
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> str:
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Tuple[str, int]]:
    try:
        full_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(full_text, chunk_size, chunk_overlap)
        return [(chunk, i) for i, chunk in enumerate(chunks)]
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def get_pdf_metadata(file_path: str) -> dict:
    try:
        with fitz.open(file_path) as doc:
            metadata = doc.metadata
        return metadata
    except Exception as e:
        raise Exception(f"Error getting PDF metadata: {str(e)}")
