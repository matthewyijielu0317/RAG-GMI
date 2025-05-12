import os
from .base_loader import BaseLoader

class TXTLoader(BaseLoader):
    """
    Loads text content from TXT files using Python's built-in file I/O.
    """
    
    def load_document(self, file_path):
        """
        Extract text from a TXT document.
        
        Args:
            file_path (str): Path to the TXT file.
            
        Returns:
            str: The extracted text content from the TXT file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return self.clean_text(text)
        
        except UnicodeDecodeError:
            # Try a different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                return self.clean_text(text)
            except Exception as e:
                raise Exception(f"Error extracting text from TXT file '{file_path}' with latin-1 encoding: {str(e)}")
        
        except Exception as e:
            raise Exception(f"Error extracting text from TXT file '{file_path}': {str(e)}")
            
    def load_document_with_metadata(self, file_path):
        """
        Extract text and metadata from a TXT document.
        
        Args:
            file_path (str): Path to the TXT file.
            
        Returns:
            tuple: (text, metadata) where text is the extracted content and 
                   metadata is a dictionary with file information.
        """
        try:
            # Get file metadata
            file_stats = os.stat(file_path)
            file_name = os.path.basename(file_path)
            
            metadata = {
                "file_name": file_name,
                "file_size": file_stats.st_size,
                "created_time": file_stats.st_ctime,
                "modified_time": file_stats.st_mtime,
                "file_extension": ".txt"
            }
            
            # Get text content
            text = self.load_document(file_path)
            
            # Add line count to metadata
            metadata["line_count"] = len(text.split('\n'))
            
            return text, metadata
        
        except Exception as e:
            raise Exception(f"Error extracting text from TXT file '{file_path}': {str(e)}") 