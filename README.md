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
│   │   └── langchain_splitter.py   # LangChain-based text splitting
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

# Modular RAG Pipeline Diagram

Document Files                                User Query
    |                                             |
    v                                             v
+------------------------+                +-------------------+
| INGESTION MODULE       |                |                   |
|                        |                |                   |
| - Document Processor   |                |                   |
| - LangChain Splitters  |                |                   |
| - Metadata Extraction  |                |                   |
+------------------------+                +-------------------+
    |                                             |
    | Chunks + Metadata                           |
    v                                             v
+---------------------------------------------------------------+
|                       RETRIEVAL MODULE                        |
|                                                               |
|  +-------------------+           +--------------------+       |
|  | Dense Retrieval   |           | Sparse Retrieval   |       |
|  | (Embeddings+FAISS)|           | (BM25)             |       |
|  +-------------------+           +--------------------+       |
|         |                                 |                   |
|         v                                 v                   |
|  +-------------------------------------------------------+   |
|  |               Result Fusion (RRF)                     |   |
|  +-------------------------------------------------------+   |
|                           |                                   |
|                           v                                   |
|  +-------------------------------------------------------+   |
|  |           Cross-Encoder Reranking                     |   |
|  +-------------------------------------------------------+   |
+---------------------------------------------------------------+
                          |
                          | Top-K Relevant Chunks
                          v
+---------------------------------------------------------------+
|                     GENERATION MODULE                         |
|                                                               |
|  +--------------------+          +--------------------+       |
|  | Retrieved Context  |          | User Query         |       |
|  +--------------------+          +--------------------+       |
|         |                                |                    |
|         v                                v                    |
|  +-------------------------------------------------------+   |
|  |               GMI Cloud API Integration               |   |
|  |                                                       |   |
|  |  - Context + Query Processing                         |   |
|  |  - LLM (Qwen/Qwen3-235B-A22B-FP8) Response Generation |   |
|  +-------------------------------------------------------+   |
+---------------------------------------------------------------+
                          |
                          v
                     Final Answer 

## Recent Updates

### System Improvements
1. **Optimized Embedding Model Loading**: Fixed duplicate model loading issue by passing the initialized model from RetrievalEngine to VectorStore.

2. **LangChain Text Splitters Integration**: 
   - Replaced custom text splitter with LangChain's more advanced text splitters
   - Added support for various splitter types (recursive, markdown)
   - Improved context preservation between chunks

3. **Tuned Parameters**: 
   - Optimized chunk size (400) and overlap (150) for better retrieval quality
   - Enhanced error handling throughout the system

4. **Dependency Updates**: 
   - Added langchain-text-splitters and tiktoken to requirements

## Modules Overview

### 1. Data Ingestion Module

Handles loading documents from various formats and preprocessing text.

**Features:**
- Multi-format document loading (PDF, TXT, DOCX)
- Text chunking with LangChain splitters for optimal chunk boundaries
- Configurable chunk size and overlap
- Metadata extraction from documents
- Batch processing of document directories
- Fallback mechanisms if LangChain dependencies aren't available

**Example Usage:**
```python
from ingestion_module import DocumentProcessor

# Initialize document processor
processor = DocumentProcessor(
    chunk_size=400,
    chunk_overlap=150,
    splitter_type="recursive"  # or "markdown" for markdown documents
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

Implements hybrid retrieval combining dense vector search, sparse keyword-based search, and cross-encoder reranking.

**Features:**
- Dense retrieval using sentence-transformers embeddings and FAISS
- Sparse retrieval using BM25 algorithm
- Result fusion using Reciprocal Rank Fusion (RRF) or linear combination
- Cross-encoder reranking for improved relevance
- Configurable retrieval parameters
- Automatic deduplication of results from different retrieval methods
- Optimized model loading to prevent redundant initialization

**Example Usage:**
```python
from retrieval_module import RetrievalEngine

# Initialize retrieval engine with both dense and sparse methods
engine = RetrievalEngine(
    embedding_model_name="all-MiniLM-L6-v2",
    enable_sparse=True,
    enable_dense=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Index documents
engine.index_documents(document_chunks, document_metadata)

# Retrieve relevant documents for a query
results = engine.retrieve(
    query="What is RAG?",
    top_k=5,
    use_reranking=True,
    fusion_method="rrf"  # or "linear"
)

# Access retrieved documents
for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.4f}")
    print(f"   Text: {result['text'][:100]}...")
```

### 3. Generation Module

Interacts with the GMI Cloud API to generate text based on a given prompt.

**Features:**
- Integration with GMI Cloud's LLM API
- Configurable generation parameters (max tokens, temperature)
- Context handling for RAG applications
- Improved error handling, including for placeholder organization IDs
- Fallback mechanisms for API interaction failures

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
    organization_id=os.getenv("GMI_ORGANIZATION_ID"),
    model_name=os.getenv("GMI_MODEL_NAME", "Qwen/Qwen3-235B-A22B-FP8")
)

# Example usage
user_question = "What is RAG?"
context_chunks = ["RAG stands for Retrieval Augmented Generation..."]

# Generate a response
response, usage = generator.generate_response(
    user_query=user_question,
    retrieved_context_chunks=context_chunks,
    system_message="Answer based only on the provided context.",
    temperature=0.7,
    max_tokens=500
)

print(response)
```

### Complete RAG Pipeline

The `RAGPipeline` class integrates all three modules into a complete RAG system.

**Features:**
- Document ingestion from files or directories
- Hybrid document retrieval with reranking
- Context-aware answer generation
- Configurable parameters for each component
- State saving and loading
- Optimized for performance with improved parameter defaults

**Example Usage:**
```python
from rag_pipeline import RAGPipeline

# Initialize the RAG pipeline with optimized parameters
pipeline = RAGPipeline(
    chunk_size=400,
    chunk_overlap=150,
    top_k=5,
    embedding_model_name="all-MiniLM-L6-v2",
    splitter_type="recursive"
)

# Ingest documents
pipeline.ingest_documents("path/to/documents")

# Answer a question
result = pipeline.answer_question(
    question="What is RAG?",
    top_k=5,
    temperature=0.7,
    max_tokens=500
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

## CLI Interface

The project provides a command-line interface for easy interaction with the RAG system:

```bash
python cli.py --docs test_docs --questions "What is RAG?"
```

### CLI Options

```
usage: cli.py [-h] --docs DOCS --questions QUESTIONS [QUESTIONS ...] [--chunk-size CHUNK_SIZE]
              [--chunk-overlap CHUNK_OVERLAP] [--top-k TOP_K] [--debug] [--temperature TEMPERATURE] 
              [--max-tokens MAX_TOKENS] [--splitter-type SPLITTER_TYPE]

RAG Pipeline CLI

options:
  -h, --help            show this help message and exit
  --docs DOCS           Path to documents directory
  --questions QUESTIONS [QUESTIONS ...]
                        Questions to answer
  --chunk-size CHUNK_SIZE
                        Chunk size for document splitting (default: 400)
  --chunk-overlap CHUNK_OVERLAP
                        Overlap between chunks (default: 150)
  --top-k TOP_K         Number of chunks to retrieve
  --debug               Enable debug mode
  --temperature TEMPERATURE
                        Generation temperature
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate
  --splitter-type SPLITTER_TYPE
                        Type of text splitter to use (recursive or markdown)
```

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

Test with different parameters:

```bash
python cli.py --docs test_docs --chunk-size 400 --chunk-overlap 150 --top-k 5 --temperature 0.3 --max-tokens 300 --questions "What is RAG?"
```

## Quick Demo

To quickly test the RAG system with example questions, use the CLI interface:

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
Indexed 137 documents with BM25
Batches: 100%|███████████████████████████████████████████| 5/5 [00:00<00:00, 13.34it/s]
Ingested 4 documents with 137 chunks

------------------------------------------------------------
Answering question: What is Joe Bruin's educational background?

Answer: Joe Bruin's educational background includes:  
- **Master of Engineering in Data Science** (Expected Dec 2025) from the University of California, Los Angeles (UCLA) with a GPA of 3.7. Coursework includes Data Mining, Natural Language Processing (NLP), Large Language Models (LLMs), and Deep Learning.  
- **Bachelor of Science in Statistics and Data Science** from UCLA with a GPA of 3.75.  
- **Bachelor of Science in Applied Mathematics** from UCLA with a GPA of 3.65. Coursework for both bachelor's degrees includes Algorithms, Data Structures, Statistics, Machine Learning, and A/B testing.  

The context does not provide additional details beyond these degrees and coursework.

Top Retrieved Context:
  [1] Score: 4.5416
      JOE BRUIN Linkedln • joebruin@gmail.com • +1 888-888-8888 EDUCATION University of California, Los Angeles Los Angeles, CA Dec. 2025 Master of Engineering in Data Science Major GPA: 3.7 Coursework: Dat...
  [2] Score: -10.8675
      demographic and education information for each county. Based on some background information, we believe that among all the variables included in this dataset, education level and race will have the st...
  [3] Score: -10.9769
      from race, education level is also highly correlated with the response variable. So, we investigate the relationship between education level and the percent voters who voted for Biden. We categorized ...

------------------------------------------------------------
Answering question: who standardizes the measurement units such as weights or other measures?

Answer: The standardization of measurement units such as weights and other measures was carried out by **Shi Huang** (the first emperor of Qin) and **Li Si**, a renowned legalistic politician and reformer. They mutually cooperated to implement these economic reforms, which included unifying currency, measurements, and horse carriages to improve trade efficiency and transportation in ancient China.

Top Retrieved Context:
  [1] Score: 7.0767
      movement was a result of the mutual cooperation of both of them. In economy, Shi Huang and Li Si determined to standardize the measurement units such as weights or other measures, the currency, and ho...
  [2] Score: 4.8149
      to the first emperor of Qin, there was another person who was as important as him. He is Li Si[3] who was a renowned legalistic politician and reformer in Chinese history. Almost all the meaningful re...
  [3] Score: -10.4952
      was the main leader of the project. In the process of building, his team encountered lots of problems which they could not solve in that time without the help of modern technologies, but they finally ...

------------------------------------------------------------
Answering question: what is the The biggest advantage of XGBoost in the report?

Answer: The biggest advantage of XGBoost mentioned in the report is its ability to perform very well without requiring extensive optimization. This is because XGBoost inherently combines feature selection, pattern learning, regularization, and cross-validation, enabling strong performance even before hyperparameter tuning.

Top Retrieved Context:
  [1] Score: 8.5254
      a bit. Finally, we found XGBoost, which is a tree based ensemble machine learning algorithm which is a scalable machine learning system for tree boosting. It uses more accurate approximations to find ...
  [2] Score: 2.9234
      expensive than when tuning hyperparameters for Random forest. The performance of XGBoost is the best in all the models we tried, but it also has very high variance. We finally decided to use XGBoost b...
  [3] Score: 2.6164
      seed can increase the performance, but a wrong seed can decrease. 2. XGBoost is a highly flexible statistical learning method, so it has a very high variance if there are no right hyperparameters to r...
```

### Additional CLI Options

The CLI interface supports several options to customize the RAG pipeline:

```bash
python cli.py --help
```

Key options:
- `--docs`: Path to documents or directory to ingest (default: "test_docs")
- `--questions`: One or more questions to ask the RAG system
- `--chunk-size`: Size of document chunks (default: 400)
- `--chunk-overlap`: Overlap between chunks (default: 150)
- `--top-k`: Number of chunks to retrieve (default: 3)
- `--temperature`: Generation temperature (default: 0.7)
- `--max-tokens`: Maximum tokens to generate (default: 500)
- `--debug`: Debug mode - only do retrieval, no generation
- `--splitter-type`: Type of text splitter to use ("recursive" or "markdown")

Example with custom parameters:
```bash
python cli.py --docs my_documents --chunk-size 400 --chunk-overlap 150 --top-k 5 --questions "What are the key findings?" --splitter-type recursive
```
