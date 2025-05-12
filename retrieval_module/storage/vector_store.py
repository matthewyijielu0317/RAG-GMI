import os
import pickle
import numpy as np
import faiss

class VectorStore:
    """
    A vector store using FAISS for efficient similarity search.
    """
    
    def __init__(self, embedding_dim=384):
        """
        Initialize a vector store.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.texts = []
        self.metadatas = []
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the FAISS index."""
        try:
            # Create a flat index (exact search)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            print(f"Initialized FAISS index with dimension {self.embedding_dim}")
        except Exception as e:
            raise Exception(f"Error initializing FAISS index: {str(e)}")
            
    def add_texts(self, texts, embeddings, metadatas=None):
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts (list): List of text chunks
            embeddings (numpy.ndarray): Array of embedding vectors
            metadatas (list, optional): List of metadata dicts for each text
            
        Returns:
            list: List of indices of the added texts
        """
        if not texts or len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts must match number of embeddings")
            
        # Convert embeddings to float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)
        
        # Add embeddings to the FAISS index
        start_index = len(self.texts)
        self.index.add(embeddings)
        
        # Store texts
        self.texts.extend(texts)
        
        # Store metadata if provided
        if metadatas:
            if len(metadatas) != len(texts):
                raise ValueError("Number of metadata items must match number of texts")
            self.metadatas.extend(metadatas)
        else:
            # Add empty metadata if not provided
            self.metadatas.extend([{} for _ in range(len(texts))])
            
        # Return indices of the added texts
        return list(range(start_index, start_index + len(texts)))
        
    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar texts.
        
        Args:
            query_embedding (numpy.ndarray): Query embedding vector
            top_k (int): Number of results to return
            
        Returns:
            list: List of dictionaries with text, score, and metadata
        """
        if self.index.ntotal == 0:
            return []
            
        # Ensure the query embedding is a 2D array (required by FAISS)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Prepare the results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Convert L2 distance to similarity score (higher is better)
            # For L2 distance, lower is better, so we negate it
            similarity_score = -distance
            
            results.append({
                "text": self.texts[idx],
                "score": similarity_score,
                "metadata": self.metadatas[idx],
                "index": idx
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
            if 0 <= idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "index": idx
                })
                
        return results
        
    def save(self, directory, name="vector_store"):
        """
        Save the vector store to disk.
        
        Args:
            directory (str): Directory to save the vector store
            name (str): Name of the vector store
            
        Returns:
            str: Path to the saved vector store
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save the FAISS index
        index_path = os.path.join(directory, f"{name}.index")
        faiss.write_index(self.index, index_path)
        
        # Save the texts and metadata
        data_path = os.path.join(directory, f"{name}.pkl")
        data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "embedding_dim": self.embedding_dim
        }
        
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
            
        print(f"Vector store saved to {directory}")
        return directory
        
    @classmethod
    def load(cls, directory, name="vector_store"):
        """
        Load a vector store from disk.
        
        Args:
            directory (str): Directory where the vector store was saved
            name (str): Name of the vector store
            
        Returns:
            VectorStore: Loaded vector store
        """
        # Load the data
        data_path = os.path.join(directory, f"{name}.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        # Create a new vector store
        vector_store = cls(embedding_dim=data["embedding_dim"])
        
        # Load the FAISS index
        index_path = os.path.join(directory, f"{name}.index")
        vector_store.index = faiss.read_index(index_path)
        
        # Set the texts and metadata
        vector_store.texts = data["texts"]
        vector_store.metadatas = data["metadatas"]
        
        print(f"Vector store loaded from {directory} with {len(vector_store.texts)} texts")
        return vector_store 