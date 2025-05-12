import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

# Try import with module prefix first, then fallback to direct import
try:
    from modular_rag_project.ingestion_module import DocumentProcessor
    from modular_rag_project.retrieval_module import RetrievalEngine
    from modular_rag_project.generation_module import GMIGenerator
except ModuleNotFoundError:
    from ingestion_module import DocumentProcessor
    from retrieval_module import RetrievalEngine
    from generation_module import GMIGenerator

class RAGPipeline:
    """
    Complete RAG pipeline that integrates data ingestion, retrieval, and generation.
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 top_k: int = 5,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 splitter_type: str = "recursive",
                 chunk_overlap: int = 100,
                 **splitter_kwargs):
        """
        Initialize the RAG pipeline.
        
        Args:
            chunk_size: Size of document chunks
            top_k: Number of chunks to retrieve
            embedding_model_name: Name of the embedding model to use
            splitter_type: Type of text splitter to use ('recursive' or 'markdown')
            chunk_overlap: Number of characters to overlap between chunks
            **splitter_kwargs: Additional parameters for the text splitter
        """
        # Load environment variables for API credentials
        load_dotenv()
        
        # Initialize modules
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            splitter_type=splitter_type,
            **splitter_kwargs
        )
        
        self.retrieval_engine = RetrievalEngine(
            embedding_model_name=embedding_model_name,
            enable_sparse=True,
            enable_dense=True
        )
        
        self.generator = GMIGenerator(
            api_key=os.getenv("GMI_API_KEY"),
            organization_id=os.getenv("GMI_ORGANIZATION_ID"),
            model_name=os.getenv("GMI_MODEL_NAME", "Qwen/Qwen3-235B-A22B-FP8")
        )
        
        # Store indexed documents
        self.documents = []
        self.document_chunks = []
        
    def ingest_documents(self, document_path, recursive=True):
        """
        Ingest documents from a file or directory.
        
        Args:
            document_path (str): Path to a file or directory
            recursive (bool): Whether to recursively process directories
            
        Returns:
            list: Processed documents information
        """
        print(f"Ingesting documents from: {document_path}")
        
        if os.path.isfile(document_path):
            # Process a single file
            result = self.document_processor.process_document(
                document_path,
                include_metadata=True,
                recursive_splitting=True
            )
            self.documents = [result]
            
        elif os.path.isdir(document_path):
            # Process a directory
            results = self.document_processor.process_directory(
                document_path,
                include_metadata=True,
                recursive_splitting=True,
                recursive_dirs=recursive
            )
            self.documents = results
            
        else:
            raise ValueError(f"Invalid document path: {document_path}")
            
        # Extract all chunks and metadata for indexing
        all_chunks = []
        all_metadata = []
        
        for doc in self.documents:
            chunks = doc["chunks"]
            metadata = doc["metadata"]
            chunk_count = len(chunks)
            
            # Create base metadata once for all chunks from this document
            base_metadata = {
                "source": metadata.get("file_path", ""),
                "file_name": metadata.get("file_name", os.path.basename(metadata.get("file_path", "")))
            }
            
            # Add to collections
            all_chunks.extend(chunks)
            all_metadata.extend([base_metadata.copy() for _ in range(chunk_count)])
        
        self.document_chunks = all_chunks
        
        # Index the chunks for retrieval
        self.retrieval_engine.index_documents(all_chunks, all_metadata)
        
        print(f"Ingested {len(self.documents)} documents with {len(all_chunks)} chunks")
        return self.documents
        
    def answer_question(self, question: str, top_k: int = 5, temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            temperature: Temperature for generation (higher = more creative)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing the answer and supporting information
        """
        # Step 1: Retrieve relevant documents
        retrieved_chunks = self.retrieval_engine.retrieve(
            query=question,
            top_k=top_k,
            fusion_method="rrf"
        )
        
        # Get just the text from the retrieval results
        retrieved_texts = [chunk["text"] for chunk in retrieved_chunks]
        
        # Step 2: Generate an answer using the retrieved context
        system_message = "You are a helpful assistant that answers questions based only on the provided context. If the information isn't in the context, say you don't know."
        
        answer, usage_stats = self.generator.generate_response(
            user_query=question,
            retrieved_context_chunks=retrieved_texts,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Prepare the result
        result = {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "usage_stats": usage_stats
        }
        
        return result
        
    def save_state(self, directory="rag_state"):
        """
        Save the state of the RAG pipeline.
        
        Args:
            directory (str): Directory to save the state
            
        Returns:
            str: Path where the state was saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save retrieval engine state
        retrieval_dir = os.path.join(directory, "retrieval")
        self.retrieval_engine.save(retrieval_dir)
        
        # Save document information
        docs_path = os.path.join(directory, "documents.json")
        
        # Create a simplified version of documents for saving
        simplified_docs = []
        for doc in self.documents:
            simplified_doc = {
                "metadata": doc["metadata"],
                "num_chunks": doc["num_chunks"]
            }
            simplified_docs.append(simplified_doc)
            
        with open(docs_path, "w") as f:
            json.dump(simplified_docs, f, indent=2)
            
        print(f"RAG pipeline state saved to {directory}")
        return directory 