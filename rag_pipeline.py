import os
import json
from dotenv import load_dotenv

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
                 chunk_size=1000, 
                 chunk_overlap=200,
                 embedding_model="all-MiniLM-L6-v2",
                 enable_sparse=True,
                 enable_dense=True,
                 fusion_method="rrf",
                 top_k=5,
                 model_name="Qwen/Qwen3-235B-A22B-FP8"):
        """
        Initialize the RAG pipeline.
        
        Args:
            chunk_size (int): Size of text chunks for splitting
            chunk_overlap (int): Overlap between chunks
            embedding_model (str): Name of the embedding model
            enable_sparse (bool): Whether to enable sparse retrieval
            enable_dense (bool): Whether to enable dense retrieval
            fusion_method (str): Method for fusing results ("rrf" or "linear")
            top_k (int): Number of documents to retrieve
            model_name (str): Name of the LLM model for generation
        """
        # Load environment variables for API credentials
        load_dotenv()
        
        # Initialize modules
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.retrieval_engine = RetrievalEngine(
            embedding_model_name=embedding_model,
            enable_sparse=enable_sparse,
            enable_dense=enable_dense
        )
        
        self.generator = GMIGenerator(
            api_key=os.getenv("GMI_API_KEY"),
            organization_id=os.getenv("GMI_ORGANIZATION_ID"),
            model_name=model_name
        )
        
        # Set retrieval parameters
        self.fusion_method = fusion_method
        self.top_k = top_k
        
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
            
            all_chunks.extend(chunks)
            
            # Create metadata for each chunk
            for _ in chunks:
                chunk_metadata = {
                    "source": metadata.get("file_path", ""),
                    "file_name": metadata.get("file_name", os.path.basename(metadata.get("file_path", "")))
                }
                all_metadata.append(chunk_metadata)
        
        self.document_chunks = all_chunks
        
        # Index the chunks for retrieval
        self.retrieval_engine.index_documents(all_chunks, all_metadata)
        
        print(f"Ingested {len(self.documents)} documents with {len(all_chunks)} chunks")
        return self.documents
        
    def answer_question(self, question, temperature=0.7, max_tokens=2000):
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question (str): The user's question
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            dict: Dictionary with generated answer and intermediate results
        """
        print(f"Answering question: {question}")
        
        # Step 1: Retrieve relevant documents
        retrieved_chunks = self.retrieval_engine.retrieve(
            query=question,
            top_k=self.top_k,
            fusion_method=self.fusion_method
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