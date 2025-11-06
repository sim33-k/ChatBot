"""
Helper functions for the Medical Chatbot application.
This module contains utility functions for loading PDFs, splitting text,
and managing embeddings.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


def load_pdf_files(data_path: str) -> List[Document]:
    """
    Load PDF files from a directory.
    
    Args:
        data_path: Path to the directory containing PDF files
        
    Returns:
        List of Document objects containing the extracted text
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to keep only essential metadata (source).
    
    Args:
        docs: List of Document objects with potentially large metadata
        
    Returns:
        List of Document objects with minimal metadata (only source)
    """
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


def text_split(minimal_docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        minimal_docs: List of Document objects to split
        chunk_size: Maximum size of each chunk (default: 500)
        chunk_overlap: Number of characters to overlap between chunks (default: 20)
        
    Returns:
        List of Document objects representing text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks


def download_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Download and initialize the embedding model.
    
    Args:
        model_name: Name of the HuggingFace model to use (default: all-MiniLM-L6-v2)
        
    Returns:
        HuggingFaceEmbeddings object ready for use
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a single string for the prompt.
    
    Args:
        docs: List of Document objects retrieved from vector store
        
    Returns:
        Concatenated string of document contents
    """
    return "\n\n".join([doc.page_content for doc in docs])
