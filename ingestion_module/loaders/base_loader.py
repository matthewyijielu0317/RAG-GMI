from abc import ABC, abstractmethod
import re

class BaseLoader(ABC):
    """
    Abstract base class for document loaders.
    All document loaders should inherit from this class.
    """
    
    def clean_text(self, text, lowercase=False):
        """
        Basic text cleaning that:
        - Keeps letters, numbers, punctuation
        - Normalizes whitespace to single spaces
        - Removes escape sequences
        - Removes leading/trailing whitespace
        - Optionally converts text to lowercase
        
        Args:
            text (str): Raw text to clean
            lowercase (bool): Whether to convert text to lowercase
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove escape sequences like \u2022, \n, \t, \r, etc.
        text = re.sub(r'\\[a-zA-Z0-9]+', ' ', text)
        
        # Normalize whitespace - replace multiple spaces/tabs/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading and trailing spaces
        text = text.strip()
        
        # Convert to lowercase if specified
        if lowercase:
            text = text.lower()
        
        return text
    
    @abstractmethod
    def load_document(self, file_path):
        """
        Load a document from the given file path.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            str: The extracted text content from the document.
        """
        pass 