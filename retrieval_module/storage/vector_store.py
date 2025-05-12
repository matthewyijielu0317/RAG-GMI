from typing import List, Dict, Any
import os
import pickle
import numpy as np
import faiss
from ..embeddings.embedding_model import EmbeddingModel

class VectorStore:
    """
    A vector store using FAISS for efficient similarity search.
    """
    
    def __init__(self, embedding_dim: int = 384, embedding_model=None, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store with FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
            embedding_model: Optional existing embedding model to reuse
            embedding_model_name: Name of the embedding model to use if creating a new one
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.metadatas = []
        # Use provided embedding model or create a new one if not provided
        self.embedding_model = embedding_model if embedding_model is not None else EmbeddingModel(model_name=embedding_model_name)
        print(f"Initialized FAISS index with dimension {embedding_dim}")
        
    def add_texts(self, texts: List[str], embeddings: List[List[float]] = None, metadatas: List[Dict] = None):
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text documents
            embeddings: Optional pre-computed embeddings
            metadatas: Optional metadata for each document
        """
        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(texts)
            
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(texts)
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))
            
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with document text, score, and metadata
        """
        # Convert query embedding to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_array, top_k)
        
        # Format results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for empty slots
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(1 / (1 + distance)),  # Convert distance to similarity score
                    "index": int(idx)
                })
                
        return results
        
    def get_by_ids(self, ids):
        """
        Retrieve texts by their indices.
        
        Args:
            ids (list): List of indices
            
        Returns:
            list: List of dictionaries with text and metadata
        """
        results = []
        for idx in ids:
            if 0 <= idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "index": idx
                })
                
        return results
        
    def save(self, directory: str, name: str = "vector_store"):
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
            name: Base name for saved files
        """
        # Save FAISS index
        faiss.write_index(self.index, f"{directory}/{name}.index")
        
        # Save documents and metadata
        import json
        with open(f"{directory}/{name}_docs.json", "w") as f:
            json.dump({
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f)
            
    @classmethod
    def load(cls, directory: str, name: str = "vector_store"):
        """
        Load vector store from disk.
        
        Args:
            directory: Directory to load from
            name: Base name for saved files
            
        Returns:
            Loaded VectorStore instance
        """
        # Load FAISS index
        index = faiss.read_index(f"{directory}/{name}.index")
        
        # Load documents and metadata
        import json
        with open(f"{directory}/{name}_docs.json", "r") as f:
            data = json.load(f)
            
        # Create new instance
        instance = cls(embedding_dim=index.d)
        instance.index = index
        instance.documents = data["documents"]
        instance.metadatas = data["metadatas"]
        
        return instance 