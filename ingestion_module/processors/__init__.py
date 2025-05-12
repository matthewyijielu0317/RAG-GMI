# Text Processors - Using LangChain Text Splitters
LANGCHAIN_AVAILABLE = False
try:
    from .langchain_splitter import LangChainTextSplitter
    # Make TextSplitter an alias for LangChainTextSplitter for backward compatibility
    TextSplitter = LangChainTextSplitter
    LANGCHAIN_AVAILABLE = True
    print("Using LangChain text splitters")
except ImportError as e:
    print(f"ERROR: LangChain text splitters not available: {e}")
    print("Please install required packages with: pip install langchain-text-splitters tiktoken")
    # Create a minimal fallback if import fails
    class TextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            print("WARNING: Using minimal fallback text splitter")
            
        def split_text(self, text):
            if not text:
                return []
            # Basic text chunking as fallback
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks
            
        def split_text_recursive(self, text):
            return self.split_text(text) 