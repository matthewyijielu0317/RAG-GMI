# Modular RAG Project

This project implements a modular Retrieval Augmented Generation (RAG) system in Python. The system consists of separate modules for data ingestion, retrieval, and generation, allowing for flexibility and independent optimization of each component.

## Project Structure

```
modular_rag_project/
├── ingestion_module/               # Document loading and text processing
│   ├── loaders/                    # Format-specific document loaders
│   │   ├── pdf_loader.py
│   │   ├── txt_loader.py
│   │   └── docx_loader.py
│   ├── processors/                 # Text processing utilities
│   │   └── text_splitter.py
│   └── document_processor.py       # Central document processing manager
├── retrieval_module/               # Document retrieval
│   ├── embeddings/                 # Vector embedding generation
│   │   └── embedding_model.py
│   ├── storage/                    # Vector storage
│   │   └── vector_store.py
│   ├── sparse_retrieval.py         # BM25 keyword-based retrieval
│   └── retrieval_engine.py         # Hybrid retrieval engine
├── generation_module/              # Text generation via LLM
│   ├── __init__.py
│   └── gmi_generator.py            # GMI Cloud API integration
├── test_docs/                      # Directory for test documents
├── rag_state/                      # Directory for saving pipeline state
├── .env                           # Environment variables for API keys
├── requirements.txt               # Project dependencies
├── rag_pipeline.py               # Complete RAG pipeline
└── cli.py                        # Command-line interface
```

## Modules Overview

### 1. Data Ingestion Module

Handles loading documents from various formats and preprocessing text.

**Features:**
- Multi-format document loading (PDF, TXT, DOCX)
- Text chunking with configurable chunk size and overlap
- Recursive text splitting that respects natural boundaries
- Metadata extraction from documents
- Batch processing of document directories

**Example Usage:**
```python
from ingestion_module import DocumentProcessor

# Initialize document processor
processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)

# Process a single document
result = processor.process_document(
    "path/to/document.pdf",
    include_metadata=True,
    recursive_splitting=True
)

# Access the extracted text and chunks
text = result["text"]
chunks = result["chunks"]
metadata = result["metadata"]

# Process an entire directory
results = processor.process_directory(
    "path/to/documents/",
    recursive_dirs=True
)
```

### 2. Retrieval Module

Implements hybrid retrieval combining dense vector search and sparse keyword-based search.

**Features:**
- Dense retrieval using sentence-transformers embeddings and FAISS
- Sparse retrieval using BM25 algorithm
- Result fusion with Reciprocal Rank Fusion (RRF) or linear combination
- Optional re-ranking of results
- Configurable retrieval parameters

**Example Usage:**
```python
from retrieval_module import RetrievalEngine

# Initialize retrieval engine with both dense and sparse methods
engine = RetrievalEngine(
    embedding_model_name="all-MiniLM-L6-v2",
    enable_sparse=True,
    enable_dense=True
)

# Index documents
engine.index_documents(document_chunks, document_metadata)

# Retrieve relevant documents for a query
results = engine.retrieve(
    query="What is RAG?",
    top_k=5,
    fusion_method="rrf"  # or "linear"
)

# Access retrieved documents
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")
```

### 3. Generation Module

Interacts with the GMI Cloud API to generate text based on a given prompt.

**Features:**
- Integration with GMI Cloud's LLM API
- Configurable generation parameters (max tokens, temperature)
- Context handling for RAG applications
- Error handling for API interactions

**Example Usage:**
```python
from generation_module import GMIGenerator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the generator
generator = GMIGenerator(
    api_key=os.getenv("GMI_API_KEY"),
    organization_id=os.getenv("GMI_ORGANIZATION_ID")
)

# Example usage
user_question = "What is RAG?"
context_chunks = ["RAG stands for Retrieval Augmented Generation..."]

# Generate a response
response, usage = generator.generate_response(
    user_query=user_question,
    retrieved_context_chunks=context_chunks,
    system_message="Answer based only on the provided context."
)

print(response)
```

### Complete RAG Pipeline

The `RAGPipeline` class integrates all three modules into a complete RAG system.

**Features:**
- Document ingestion from files or directories
- Hybrid document retrieval
- Context-aware answer generation
- Configurable parameters for each component
- State saving and loading

**Example Usage:**
```python
from rag_pipeline import RAGPipeline

# Initialize the RAG pipeline
pipeline = RAGPipeline(
    chunk_size=300,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    enable_sparse=True,
    enable_dense=True,
    fusion_method="rrf",
    top_k=3
)

# Ingest documents
pipeline.ingest_documents("path/to/documents")

# Answer a question
result = pipeline.answer_question(
    "What is RAG?",
    temperature=0.7,
    max_tokens=1000
)

# Access the answer and retrieved context
answer = result["answer"]
retrieved_chunks = result["retrieved_chunks"]
usage_stats = result["usage_stats"]

# Save the pipeline state
pipeline.save_state("rag_state")
```

## Setup Instructions

1. **Create a Python virtual environment**:

   ```bash
   # Create virtual environment
   python -m venv rag_env
   
   # Activate virtual environment (macOS/Linux)
   source rag_env/bin/activate
   
   # Activate virtual environment (Windows)
   rag_env\Scripts\activate
   ```

2. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your environment variables**:

   Edit the `.env` file and replace the placeholders with your actual credentials:
   ```
   GMI_API_KEY="YOUR_GMI_API_KEY_HERE"
   GMI_ORGANIZATION_ID="YOUR_GMI_ORGANIZATION_ID_HERE"
   ```
   
   **Important**: 
   - Make sure to add `.env` to your `.gitignore` file to prevent exposing your credentials
   - If you don't have a `.gitignore` file, create one and add `.env` to it

## Testing

The RAG system can be tested using the CLI interface. First, ensure you have some test documents in the `test_docs` directory (PDF, TXT, DOCX formats are supported).

### Basic Testing

Run the CLI with default parameters:

```bash
python cli.py --docs test_docs --questions "What is RAG?"
```

### Debug Mode Testing

To test only the retrieval component without generation:

```bash
python cli.py --docs test_docs --questions "What is RAG?" --debug
```

### Testing with Custom Parameters

Test with different chunk sizes and retrieval parameters:

```bash
python cli.py --docs test_docs --chunk-size 500 --chunk-overlap 100 --top-k 5 --questions "What is RAG?"
```

## API Parameters

The `generate_response` method accepts the following parameters:

- `user_query` (str): The user's question or prompt
- `retrieved_context_chunks` (list): List of text chunks providing context for the answer
- `system_message` (str, optional): System message to guide LLM behavior
- `max_tokens` (int, optional): Maximum number of tokens in the response (default: 2000)
- `temperature` (float, optional): Controls randomness in output (default: 0.7)

## Quick Demo

To quickly test the RAG system with some example questions, you can use the CLI interface:

```bash
python cli.py --docs test_docs --questions "What is Joe Bruin's educational background?" "who standardizes the measurement units such as weights or other measures?" "what is the The biggest advantage of XGBoost in the report?"
```

Example output:
```
==================================================
MODULAR RAG PIPELINE
==================================================
Loaded model 'all-MiniLM-L6-v2' with dimension 384
Initialized FAISS index with dimension 384
Dense retrieval enabled with model: all-MiniLM-L6-v2
Sparse retrieval enabled with BM25
Ingesting documents from: test_docs
Indexed 346 documents with BM25
Ingested 4 documents with 346 chunks

------------------------------------------------------------
Answering question: What is Joe Bruin's educational background?

Top Retrieved Context Chunks:
  [1] Score: 0.0333
      JOE BRUIN Linkedln • joebruin@gmail.com • +1 888-888-8888 EDUCATION University of California, Los Angeles Los Angeles, CA Dec. 2025 Master of Engineering in Data Science Major GPA: 3.7

Answer: Joe Bruin's educational background includes:  
- Master of Engineering in Data Science from UCLA (expected Dec. 2025)
- GPA: 3.7
- Coursework includes Data Mining, NLP, LLMs, and Deep Learning

------------------------------------------------------------
Answering question: who standardizes the measurement units such as weights or other measures?

Top Retrieved Context Chunks:
  [1] Score: 0.0333
      result of the mutual cooperation of both of them. In economy, Shi Huang and Li Si determined to standardize the measurement units such as weights or other measures...

Answer: Shi Huang and Li Si standardized the measurement units such as weights and other measures.

------------------------------------------------------------
Answering question: what is the The biggest advantage of XGBoost in the report?

Top Retrieved Context Chunks:
  [1] Score: 0.0333
      curate approximations to find the best tree model. The biggest advantage of XGBoost is that it can perform very well without any optimization because it combines feature selection, pattern learning, regularization, and cross validation.

Answer: The biggest advantage of XGBoost is that it can perform very well without any optimization because it combines feature selection, pattern learning, regularization, and cross validation.
```

### Additional CLI Options

The CLI interface supports several options to customize the RAG pipeline:

```bash
python cli.py --help
```

Key options:
- `--docs`: Path to documents or directory to ingest (default: "test_docs")
- `--questions`: One or more questions to ask the RAG system
- `--chunk-size`: Size of document chunks (default: 300)
- `--chunk-overlap`: Overlap between chunks (default: 50)
- `--top-k`: Number of chunks to retrieve (default: 3)
- `--temperature`: Generation temperature (default: 0.7)
- `--max-tokens`: Maximum tokens to generate (default: 500)
- `--debug`: Debug mode - only do retrieval, no generation

Example with custom parameters:
```bash
python cli.py --docs my_documents --chunk-size 500 --top-k 5 --questions "What are the key findings?"
```