import fitz  # PyMuPDF
from .base_loader import BaseLoader

class PDFLoader(BaseLoader):
    """
    Loads text content from PDF files using PyMuPDF (fitz).
    """
    
    def load_document(self, file_path):
        """
        Extract text from a PDF document.
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            str: The extracted text content from the PDF.
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text
                text += "\n\n"  # Add separation between pages
            
            doc.close()
            return self.clean_text(text)
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF '{file_path}': {str(e)}")
            
    def load_document_with_metadata(self, file_path):
        """
        Extract text and metadata from a PDF document.
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            tuple: (text, metadata) where text is the extracted content and 
                   metadata is a dictionary with info like page numbers, title, etc.
        """
        try:
            doc = fitz.open(file_path)
            full_text = ""
            pages_content = []
            
            # Extract document metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "total_pages": len(doc)
            }
            
            # Extract text from each page with page numbers
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                clean_page_text = self.clean_text(page_text)
                
                full_text += clean_page_text
                full_text += "\n\n"
                
                pages_content.append({
                    "page_number": page_num + 1,
                    "content": clean_page_text
                })
            
            # Add pages content to metadata
            metadata["pages"] = pages_content
            
            doc.close()
            return self.clean_text(full_text), metadata
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF '{file_path}': {str(e)}") 