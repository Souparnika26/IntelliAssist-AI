"""
Document Loader Module for IntelliAssist AI.
Handles multi-format parsing (PDF, DOCX, TXT) directly from Streamlit uploads
as well as standard local disk paths.
Ensures page-level metadata is preserved for citation-backed RAG.
"""

import os
from typing import List, Dict, Any, Union
from pypdf import PdfReader
from docx import Document


class DocumentLoader:
    """Handles extraction of text and metadata from various file formats."""

    @staticmethod
    def _get_filename(file_source: Any) -> str:
        """Safely extracts the filename whether source is a path or Streamlit UploadedFile."""
        if isinstance(file_source, str):
            return os.path.basename(file_source)
        return getattr(file_source, "name", "uploaded_document")

    @classmethod
    def load_pdf(cls, file: Union[Any, str]) -> List[Dict[str, Any]]:
        """Extracts text from PDF with page-by-page metadata."""
        documents = []
        filename = cls._get_filename(file)
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "content": text.strip(),
                        "metadata": {
                            "source": filename,
                            "page": i + 1,
                            "file_type": "pdf"
                        }
                    })
        except Exception as e:
            print(f"Error parsing PDF {filename}: {e}")
        return documents

    @classmethod
    def load_docx(cls, file: Union[Any, str]) -> List[Dict[str, Any]]:
        """Extracts text from DOCX files."""
        documents = []
        filename = cls._get_filename(file)
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            doc = Document(file)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)

            if full_text:
                documents.append({
                    "content": full_text,
                    "metadata": {
                        "source": filename,
                        "page": 1,
                        "file_type": "docx"
                    }
                })
        except Exception as e:
            print(f"Error parsing DOCX {filename}: {e}")
        return documents

    @classmethod
    def load_txt(cls, file: Union[Any, str]) -> List[Dict[str, Any]]:
        """Extracts text from plain TXT files or buffers."""
        documents = []
        filename = cls._get_filename(file)
        try:
            if hasattr(file, "seek") and hasattr(file, "read"):
                file.seek(0)
                raw_data = file.read()
                text = raw_data.decode("utf-8", errors="replace") if isinstance(raw_data, bytes) else raw_data
            else:
                with open(str(file), "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()

            if text.strip():
                documents.append({
                    "content": text.strip(),
                    "metadata": {
                        "source": filename,
                        "page": 1,
                        "file_type": "txt"
                    }
                })
        except Exception as e:
            print(f"Error parsing TXT {filename}: {e}")
        return documents

    @classmethod
    def process_upload(cls, uploaded_file: Union[Any, str]) -> List[Dict[str, Any]]:
        """Routes the uploaded file or file path to the appropriate parser."""
        filename = cls._get_filename(uploaded_file).lower()

        if filename.endswith(".pdf"):
            return cls.load_pdf(uploaded_file)
        elif filename.endswith(".docx"):
            return cls.load_docx(uploaded_file)
        elif filename.endswith(".txt"):
            return cls.load_txt(uploaded_file)
        else:
            return []