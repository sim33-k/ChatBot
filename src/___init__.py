"""
A RAG-based information chatbot using LangChain, Pinecone, and Google Gemini.
"""

from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings,
    format_docs
)

from src.prompt import (
    get_medical_prompt,
    get_detailed_prompt,
    get_simple_prompt,
    MEDICAL_PROMPT_TEMPLATE,
    DETAILED_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE
)

from src.store_index import (
    PineconeIndexManager,
    create_pinecone_index
)

__version__ = "0.1.0"

__all__ = [
    # Helper functions
    "load_pdf_files",
    "filter_to_minimal_docs",
    "text_split",
    "download_embeddings",
    "format_docs",
    
    # Prompt functions
    "get_medical_prompt",
    "get_detailed_prompt",
    "get_simple_prompt",
    
    # Prompt templates
    "MEDICAL_PROMPT_TEMPLATE",
    "DETAILED_PROMPT_TEMPLATE",
    "SIMPLE_PROMPT_TEMPLATE",
    
    # Pinecone index management
    "PineconeIndexManager",
    "create_pinecone_index",
]
