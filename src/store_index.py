"""
Pinecone Vector Store Management for Medical Chatbot.
This module handles the creation, uploading, and retrieval of document embeddings
using Pinecone vector database.
"""

import os
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


class PineconeIndexManager:
    """
    Manages Pinecone index creation, document uploading, and retrieval operations.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        index_name: str = "medical-chatbot",
        dimension: int = 384,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize the Pinecone Index Manager.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index
            dimension: Dimension of the embedding vectors (384 for all-MiniLM-L6-v2)
            metric: Similarity metric to use (cosine, euclidean, dotproduct)
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region for the index
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key must be provided or set in PINECONE_API_KEY environment variable")
        
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        self.docsearch = None
    
    def create_index(self, force: bool = False) -> None:
        """
        Create a new Pinecone index if it doesn't exist.
        
        Args:
            force: If True, delete existing index and create a new one
        """
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name in existing_indexes:
            if force:
                print(f"Deleting existing index: {self.index_name}")
                self.pc.delete_index(self.index_name)
            else:
                print(f"Index '{self.index_name}' already exists. Skipping creation.")
                self.index = self.pc.Index(self.index_name)
                return
        
        print(f"Creating new index: {self.index_name}")
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(
                cloud=self.cloud,
                region=self.region
            )
        )
        print(f"Index '{self.index_name}' created successfully!")
        self.index = self.pc.Index(self.index_name)
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary containing index statistics
        """
        if not self.index:
            self.index = self.pc.Index(self.index_name)
        
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get('total_vector_count', 0),
            "dimension": stats.get('dimension', self.dimension),
            "index_fullness": stats.get('index_fullness', 0.0),
            "namespaces": stats.get('namespaces', {})
        }
    
    def upload_documents(
        self, 
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
        force_upload: bool = False
    ) -> PineconeVectorStore:
        """
        Upload documents to Pinecone index.
        
        Args:
            documents: List of Document objects to upload
            embeddings: HuggingFaceEmbeddings instance for embedding generation
            force_upload: If True, upload even if index already has vectors
            
        Returns:
            PineconeVectorStore instance
        """
        if not self.index:
            self.index = self.pc.Index(self.index_name)
        
        stats = self.get_index_stats()
        total_vectors = stats['total_vectors']
        
        if total_vectors > 0 and not force_upload:
            print(f"Index already contains {total_vectors} vectors. Skipping upload to avoid duplicates.")
            print("Loading existing index...")
            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=embeddings
            )
            print("Connected to existing index.")
        else:
            if total_vectors > 0:
                print(f"Force upload enabled. Index currently has {total_vectors} vectors.")
            print(f"Uploading {len(documents)} document chunks to Pinecone...")
            self.docsearch = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=self.index_name
            )
            print(f"Successfully uploaded {len(documents)} document chunks to Pinecone!")
        
        return self.docsearch
    
    def load_existing_index(
        self, 
        embeddings: HuggingFaceEmbeddings
    ) -> PineconeVectorStore:
        """
        Load an existing Pinecone index.
        
        Args:
            embeddings: HuggingFaceEmbeddings instance for query embedding
            
        Returns:
            PineconeVectorStore instance
        """
        print(f"Loading existing Pinecone index: {self.index_name}")
        self.docsearch = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=embeddings
        )
        print("Successfully loaded existing index.")
        return self.docsearch
    
    def get_retriever(
        self, 
        search_type: str = "similarity",
        k: int = 3,
        **kwargs
    ):
        """
        Get a retriever from the vector store.
        
        Args:
            search_type: Type of search (similarity, mmr, similarity_score_threshold)
            k: Number of documents to retrieve
            **kwargs: Additional search parameters
            
        Returns:
            Retriever instance
        """
        if not self.docsearch:
            raise ValueError("Vector store not initialized. Call upload_documents or load_existing_index first.")
        
        search_kwargs = {"k": k}
        search_kwargs.update(kwargs)
        
        return self.docsearch.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_index(self) -> None:
        """
        Delete the Pinecone index.
        
        Warning: This will permanently delete all vectors in the index.
        """
        print(f"Deleting index: {self.index_name}")
        self.pc.delete_index(self.index_name)
        print(f"Index '{self.index_name}' deleted successfully.")
        self.index = None
        self.docsearch = None


def create_pinecone_index(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    index_name: str = "medical-chatbot",
    dimension: int = 384,
    api_key: Optional[str] = None,
    force_recreate: bool = False,
    force_upload: bool = False
) -> tuple[PineconeIndexManager, PineconeVectorStore]:
    """
    Convenience function to create and populate a Pinecone index.
    
    Args:
        documents: List of Document objects to upload
        embeddings: HuggingFaceEmbeddings instance
        index_name: Name of the Pinecone index
        dimension: Dimension of the embedding vectors
        api_key: Pinecone API key (optional if set in env)
        force_recreate: If True, delete and recreate the index
        force_upload: If True, upload documents even if index has vectors
        
    Returns:
        Tuple of (PineconeIndexManager, PineconeVectorStore)
    """
    manager = PineconeIndexManager(
        api_key=api_key,
        index_name=index_name,
        dimension=dimension
    )
    
    # Create index
    manager.create_index(force=force_recreate)
    
    # Upload documents
    docsearch = manager.upload_documents(
        documents=documents,
        embeddings=embeddings,
        force_upload=force_upload
    )
    
    return manager, docsearch
