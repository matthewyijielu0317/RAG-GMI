import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """
    Handles text embedding using sentence-transformers models.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use
                              Defaults to 'all-MiniLM-L6-v2' which is fast and has
                              good performance for semantic search.
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension from the model
            sample_text = "Sample text to determine embedding dimension."
            sample_embedding = self.model.encode(sample_text)
            self.embedding_dimension = len(sample_embedding)
            print(f"Loaded model '{self.model_name}' with dimension {self.embedding_dimension}")
        except Exception as e:
            raise Exception(f"Error loading embedding model '{self.model_name}': {str(e)}")
            
    def embed_text(self, text):
        """
        Generate an embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
            
        if self.model is None:
            self._load_model()
            
        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error embedding text: {str(e)}")
            
    def embed_batch(self, texts, batch_size=32, show_progress=True):
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (list): List of texts to embed
            batch_size (int): Batch size for embedding
            show_progress (bool): Whether to show progress bar
            
        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list of strings")
            
        if self.model is None:
            self._load_model()
            
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=show_progress
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Error embedding batch of texts: {str(e)}")
            
    def similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First embedding vector
            embedding2 (numpy.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score (between -1 and 1)
        """
        # Normalize the embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        return np.dot(embedding1_norm, embedding2_norm)
        
    def batch_similarity(self, query_embedding, document_embeddings):
        """
        Calculate cosine similarities between a query embedding and multiple document embeddings.
        
        Args:
            query_embedding (numpy.ndarray): Query embedding vector
            document_embeddings (numpy.ndarray): Array of document embedding vectors
            
        Returns:
            numpy.ndarray: Array of similarity scores
        """
        # Normalize the query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize the document embeddings
        document_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        normalized_document_embeddings = document_embeddings / document_norms
        
        # Calculate similarities
        similarities = np.dot(normalized_document_embeddings, query_embedding)
        
        return similarities 