from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from typing import List, Optional, Dict, Any

class LangChainTextSplitter:
    """
    A wrapper for LangChain's various text splitters, providing a consistent interface
    with the existing TextSplitter class.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200, 
                 splitter_type: str = "recursive",
                 **kwargs):
        """
        Initialize the LangChain text splitter wrapper.
        
        Args:
            chunk_size: Target size of chunks (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            splitter_type: Type of LangChain splitter to use ('recursive', 'markdown')
            **kwargs: Additional parameters for the specific LangChain splitter
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type.lower()
        self.kwargs = kwargs
        self._initialize_splitter()
        
    def _initialize_splitter(self):
        """Initialize the appropriate LangChain splitter based on type."""
        if self.splitter_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                **self.kwargs
            )
        elif self.splitter_type == "markdown":
            # For markdown, we need headers configuration
            headers_to_split_on = self.kwargs.pop("headers_to_split_on", [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ])
            
            # Create a markdown splitter with headers config
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            
            # Combine with a recursive splitter to handle chunk sizes
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown splitter type: {self.splitter_type}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using the appropriate LangChain splitter.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        try:    
            return self.splitter.split_text(text)
        except Exception as e:
            # Fallback to simpler approach if there's an error
            print(f"Error using LangChain splitter: {e}, falling back to basic splitter")
            # Basic text chunking as fallback
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks
        
    def split_text_recursive(self, text: str) -> List[str]:
        """
        Maintain API compatibility with existing TextSplitter.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        return self.split_text(text)
        
    @classmethod
    def available_splitters(cls) -> List[str]:
        """Return a list of available splitter types."""
        return ["recursive", "markdown"]
        
    def get_splitter_info(self) -> Dict[str, Any]:
        """Return information about the current splitter configuration."""
        return {
            "type": self.splitter_type,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "additional_params": self.kwargs
        } 