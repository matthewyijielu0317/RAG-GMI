from .embeddings.embedding_model import EmbeddingModel
from .storage.vector_store import VectorStore
from .sparse_retrieval import SparseRetriever
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder

class RetrievalEngine:
    """
    Main retrieval engine that combines dense and sparse retrieval methods.
    """
    
    def __init__(self, 
                 embedding_model_name="all-MiniLM-L6-v2", 
                 enable_sparse=True,
                 enable_dense=True,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the retrieval engine.
        
        Args:
            embedding_model_name (str): Name of the embedding model to use
            enable_sparse (bool): Whether to enable sparse retrieval (BM25)
            enable_dense (bool): Whether to enable dense retrieval (vector search)
            reranker_model: Name of the cross-encoder model for reranking
        """
        self.enable_sparse = enable_sparse
        self.enable_dense = enable_dense
        
        # Ensure at least one retrieval method is enabled
        if not (enable_sparse or enable_dense):
            raise ValueError("At least one retrieval method must be enabled (sparse or dense)")
            
        # Initialize embedding model if dense retrieval is enabled
        if self.enable_dense:
            self.embedding_model = EmbeddingModel(model_name=embedding_model_name)
            self.vector_store = VectorStore(
                embedding_dim=self.embedding_model.embedding_dimension,
                embedding_model=self.embedding_model  # Pass the already initialized embedding model
            )
            print(f"Dense retrieval enabled with model: {embedding_model_name}")
        else:
            self.embedding_model = None
            self.vector_store = None
            
        # Initialize sparse retriever if sparse retrieval is enabled
        if self.enable_sparse:
            self.sparse_retriever = SparseRetriever()
            print("Sparse retrieval enabled with BM25")
        else:
            self.sparse_retriever = None
            
        # Initialize cross-encoder model for reranking
        self.reranker = CrossEncoder(reranker_model)
        
    def index_documents(self, documents, metadatas=None):
        """
        Index a list of documents for retrieval.
        
        Args:
            documents (list): List of document texts
            metadatas (list, optional): List of metadata dictionaries for each document
            
        Returns:
            self: For method chaining
        """
        # Index documents for sparse retrieval
        if self.enable_sparse:
            self.sparse_retriever.index_documents(documents)
            
        # Index documents for dense retrieval
        if self.enable_dense:
            # Generate embeddings for all documents
            embeddings = self.embedding_model.embed_batch(documents)
            
            # Add documents to vector store
            self.vector_store.add_texts(documents, embeddings, metadatas)
            
        return self
        
    def retrieve(self, 
                query: str, 
                top_k: int = 5,
                use_reranking: bool = True,
                fusion_method: str = "rrf",
                alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid retrieval and optional reranking.
        
        Args:
            query: The search query
            top_k: Number of final results to return
            use_reranking: Whether to use cross-encoder reranking
            fusion_method: Method for fusion ("rrf" or "linear")
            alpha: Weight for dense scores when using linear combination
            
        Returns:
            List of retrieved documents with their scores
        """
        sparse_results = []
        dense_results = []
        
        # Get sparse retrieval results
        if self.enable_sparse:
            sparse_results = self.sparse_retriever.search(query, top_k=top_k*2)
            
        # Get dense retrieval results
        if self.enable_dense:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Retrieve similar documents
            dense_results = self.vector_store.search(query_embedding, top_k=top_k*2)
            
        # If only one method is enabled, return its results
        if not self.enable_sparse and self.enable_dense:
            results = dense_results
        elif self.enable_sparse and not self.enable_dense:
            results = sparse_results
        else:
            # Combine results using fusion
            results = self._fuse_results(
                sparse_results, 
                dense_results, 
                method=fusion_method,
                alpha=alpha,
                top_k=top_k*2
            )
        
        # Apply reranking if enabled
        if use_reranking and results:
            reranked_results = self.rerank_results(query, results, top_k)
            return reranked_results
            
        # Otherwise return top_k results
        return results[:top_k]
        
    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank results using a cross-encoder model.
        
        Args:
            query: The original query
            results: The initial retrieval results
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results
        """
        if not results:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [(query, doc['text']) for doc in results]
        
        # Get relevance scores from cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Combine scores with original results
        scored_results = list(zip(results, scores))
        
        # Sort by cross-encoder scores
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results with updated scores
        reranked = []
        for doc, score in scored_results[:top_k]:
            doc['score'] = float(score)  # Convert to float for JSON serialization
            reranked.append(doc)
            
        return reranked
        
    def save(self, directory):
        """
        Save the retrieval engine state.
        
        Args:
            directory (str): Directory to save the state
            
        Returns:
            str: Path where the state was saved
        """
        # Save vector store if dense retrieval is enabled
        if self.enable_dense and self.vector_store:
            self.vector_store.save(directory, name="vector_store")
            
        # For now, we don't save the sparse retriever state
        # In a real implementation, you might want to save the BM25 model
        
        return directory 

    def _fuse_results(self, sparse_results, dense_results, method="rrf", alpha=0.5, top_k=5):
        """
        Fuse results from sparse and dense retrieval.
        
        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval
            method: Fusion method ("rrf" or "linear")
            alpha: Weight for dense scores when using linear combination
            top_k: Number of results to return after fusion
            
        Returns:
            List of fused and re-ranked results
        """
        # Create a dictionary to store combined results
        combined_dict = {}
        
        if method == "rrf":
            # Reciprocal Rank Fusion
            # Constant to prevent division by zero and reduce the impact of high rankings
            k = 60
            
            # Process dense results
            for rank, result in enumerate(dense_results):
                doc_id = result["index"]
                if doc_id not in combined_dict:
                    combined_dict[doc_id] = {
                        "text": result["text"],
                        "metadata": result.get("metadata", {}),
                        "score": 0,
                        "index": doc_id
                    }
                # Add RRF score from dense retrieval
                combined_dict[doc_id]["score"] += 1.0 / (rank + k)
                
            # Process sparse results
            for rank, result in enumerate(sparse_results):
                doc_id = result["index"]
                if doc_id not in combined_dict:
                    combined_dict[doc_id] = {
                        "text": result["text"],
                        "metadata": {},  # Sparse results may not have metadata
                        "score": 0,
                        "index": doc_id
                    }
                # Add RRF score from sparse retrieval
                combined_dict[doc_id]["score"] += 1.0 / (rank + k)
                
        elif method == "linear":
            # Linear combination of scores
            
            # Normalize sparse scores
            max_sparse_score = max([r["score"] for r in sparse_results]) if sparse_results else 1
            
            # Normalize dense scores
            max_dense_score = max([r["score"] for r in dense_results]) if dense_results else 1
            
            # Process dense results
            for result in dense_results:
                doc_id = result["index"]
                if doc_id not in combined_dict:
                    combined_dict[doc_id] = {
                        "text": result["text"],
                        "metadata": result.get("metadata", {}),
                        "score": 0,
                        "index": doc_id
                    }
                # Add normalized and weighted dense score
                if max_dense_score > 0:
                    combined_dict[doc_id]["score"] += alpha * (result["score"] / max_dense_score)
                
            # Process sparse results
            for result in sparse_results:
                doc_id = result["index"]
                if doc_id not in combined_dict:
                    combined_dict[doc_id] = {
                        "text": result["text"],
                        "metadata": {},
                        "score": 0,
                        "index": doc_id
                    }
                # Add normalized and weighted sparse score
                if max_sparse_score > 0:
                    combined_dict[doc_id]["score"] += (1 - alpha) * (result["score"] / max_sparse_score)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        # Convert combined dictionary to a list and sort by score
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k results
        return combined_results[:top_k] 