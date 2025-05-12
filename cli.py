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
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    # Main arguments
    parser.add_argument("--docs", required=True, help="Path to documents directory")
    parser.add_argument("--questions", nargs="+", required=True, help="Questions to answer")
    
    # Processing parameters
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for document splitting")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for document splitting")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    
    # Text splitting options
    parser.add_argument("--splitter-type", choices=["recursive", "markdown"], default="recursive", 
                        help="Text splitter type to use")
    
    # Generation parameters
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("MODULAR RAG PIPELINE")
    print("="*50 + "\n")
    
    # Initialize pipeline
    splitter_kwargs = {"chunk_overlap": args.chunk_overlap}
    
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        splitter_type=args.splitter_type,
        **splitter_kwargs
    )
    
    # Ingest documents
    pipeline.ingest_documents(args.docs)
    
    # Answer questions
    for question in args.questions:
        print("\n" + "-"*60)
        print(f"Answering question: {question}")
        
        if args.debug:
            # In debug mode, only do retrieval without generation
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
                question=question,
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            print("\nAnswer:", result["answer"])
            
            # Optional: Display retrieved chunks in normal mode too
            print("\nTop Retrieved Context:")
            for i, chunk in enumerate(result["retrieved_chunks"][:3], 1):
                print(f"  [{i}] Score: {chunk['score']:.4f}")
                preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                print(f"      {preview}")
                
if __name__ == "__main__":
    main() 