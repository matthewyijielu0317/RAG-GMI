import re
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

class SparseRetriever:
    """
    Implements sparse retrieval using the BM25 algorithm.
    """
    
    def __init__(self):
        """Initialize the sparse retriever."""
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
        self._ensure_nltk_resources()
        self.stopwords = set(stopwords.words('english'))
        
    def _ensure_nltk_resources(self):
        """Ensure that the required NLTK resources are downloaded."""
        try:
            # Try to tokenize a sample text
            word_tokenize("test")
        except LookupError:
            # Download the required resources if not available
            print("Downloading NLTK resources...")
            nltk.download('punkt')
            
        try:
            # Try to access stopwords
            stopwords.words('english')
        except LookupError:
            # Download stopwords if not available
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
            
    def preprocess_text(self, text):
        """
        Preprocess text for indexing or searching.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: List of tokens
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]
        
        return tokens
        
    def index_documents(self, documents):
        """
        Index a list of document texts using BM25.
        
        Args:
            documents (list): List of document texts
            
        Returns:
            self: For method chaining
        """
        self.corpus = documents
        self.tokenized_corpus = [self.preprocess_text(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"Indexed {len(documents)} documents with BM25")
        return self
        
    def search(self, query, top_k=5):
        """
        Search for relevant documents using BM25.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of dictionaries with document text, score, and index
        """
        # Check if the index is empty
        if not self.corpus or not self.bm25:
            return []
            
        # Preprocess the query
        tokenized_query = self.preprocess_text(query)
        
        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get the top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Prepare the results
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:  # Only include results with non-zero scores
                results.append({
                    "text": self.corpus[idx],
                    "score": float(score),  # Convert numpy type to native Python type
                    "index": int(idx)       # Convert numpy type to native Python type
                })
                
        return results
        
    def get_document(self, index):
        """
        Get a document by its index.
        
        Args:
            index (int): Document index
            
        Returns:
            str: Document text
        """
        if 0 <= index < len(self.corpus):
            return self.corpus[index]
        else:
            return None 