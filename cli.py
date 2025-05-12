#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv

# Try import with module prefix first, then fallback to direct import
try:
    from modular_rag_project.rag_pipeline import RAGPipeline
except ModuleNotFoundError:
    from rag_pipeline import RAGPipeline

def main():
    """
    Command-line interface for the RAG pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the RAG pipeline with custom parameters")
    parser.add_argument("--docs", type=str, help="Path to documents or directory to ingest", default="test_docs")
    parser.add_argument("--questions", type=str, nargs="+", help="Questions to ask the RAG system")
    parser.add_argument("--chunk-size", type=int, help="Size of document chunks", default=300)
    parser.add_argument("--chunk-overlap", type=int, help="Overlap between chunks", default=50)
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve", default=3)
    parser.add_argument("--temperature", type=float, help="Generation temperature", default=0.7)
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate", default=500)
    parser.add_argument("--debug", action="store_true", help="Debug mode - only do retrieval, no generation")
    args = parser.parse_args()
    
    # Check for API keys
    load_dotenv()
    if not os.getenv("GMI_API_KEY") or os.getenv("GMI_API_KEY") == "YOUR_GMI_API_KEY_HERE":
        print("Error: Please set your GMI API key in the .env file")
        return
        
    print("\n" + "=" * 50)
    print("MODULAR RAG PIPELINE")
    print("=" * 50)
    
    # Initialize the RAG pipeline with command line arguments or defaults
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model="all-MiniLM-L6-v2",
        enable_sparse=True,
        enable_dense=True,
        fusion_method="rrf",
        top_k=args.top_k
    )
    
    # Check if document directory exists
    doc_path = args.docs
    if not os.path.exists(doc_path):
        print(f"Error: Path does not exist: {doc_path}")
        return
        
    if os.path.isdir(doc_path) and not os.listdir(doc_path):
        print(f"No files found in {doc_path}. Please add some documents.")
        return
        
    # Ingest documents
    pipeline.ingest_documents(doc_path)
    
    # Use provided questions or defaults
    questions = args.questions if args.questions else [
        "What is RAG?",
        "How does modular RAG architecture improve system performance?",
        "Explain the three main components of RAG."
    ]
    
    # Answer each question
    for question in questions:
        print("\n" + "-" * 60)
        
        if args.debug:
            # In debug mode, only do retrieval without generation
            print(f"\nQuestion: {question}")
            retrieved_chunks = pipeline.retrieval_engine.retrieve(
                query=question,
                top_k=args.top_k,
                fusion_method="rrf"
            )
            
            print("\nTop Retrieved Context Chunks:")
            for i, chunk in enumerate(retrieved_chunks[:3]):  # Show top 3 chunks
                print(f"  [{i+1}] Score: {chunk['score']:.4f}")
                preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                print(f"      {preview}")
        else:
            # Normal mode with generation
            result = pipeline.answer_question(
                question, 
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            print(f"\nQuestion: {result['question']}")
            
            print("\nTop Retrieved Context Chunks:")
            for i, chunk in enumerate(result['retrieved_chunks'][:3]):  # Show top 3 chunks
                print(f"  [{i+1}] Score: {chunk['score']:.4f}")
                preview = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                print(f"      {preview}")
                
            print("\nAnswer:")
            print(result['answer'])
    
    # Save the pipeline state
    pipeline.save_state()

if __name__ == "__main__":
    main() 