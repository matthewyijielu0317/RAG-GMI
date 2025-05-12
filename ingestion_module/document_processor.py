import os
from .loaders import PDFLoader, TXTLoader, DOCXLoader
from .processors import TextSplitter, LANGCHAIN_AVAILABLE

if LANGCHAIN_AVAILABLE:
    from .processors import LangChainTextSplitter

class DocumentProcessor:
    """
    Central manager for document processing.
    Handles loading documents from different formats and processing their text.
    """
    
    def __init__(self, chunk_size=1000, 
                 splitter_type="recursive", **splitter_kwargs):
        """
        Initialize the document processor with text splitter settings.
        
        Args:
            chunk_size (int): Target size of each text chunk (in characters)
            splitter_type (str): Type of LangChain splitter to use ('recursive' or 'markdown')
            **splitter_kwargs: Additional parameters for the text splitter (e.g., chunk_overlap)
        """
        # Initialize loaders for different document formats
        self.loaders = {
            ".pdf": PDFLoader(),
            ".txt": TXTLoader(),
            ".docx": DOCXLoader()
        }

        # Initialize text splitter
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            splitter_type=splitter_type,
            **splitter_kwargs
        )
        
    def process_document(self, file_path, include_metadata=False, recursive_splitting=True):
        """
        Process a document: load it and split into chunks.
        
        Args:
            file_path (str): Path to the document file
            include_metadata (bool): Whether to include document metadata
            recursive_splitting (bool): Whether to use recursive text splitting
            
        Returns:
            dict: A dictionary containing the original text, chunks, and optional metadata
        """
        # Get file extension
        _, file_extension = os.path.splitext(file_path.lower())
        
        # Check if we have a loader for this file type
        if file_extension not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Get the appropriate loader
        loader = self.loaders[file_extension]
        
        # Load the document
        if include_metadata:
            full_text, metadata = loader.load_document_with_metadata(file_path)
        else:
            full_text = loader.load_document(file_path)
            metadata = {"file_path": file_path}
        
        # Split the text into chunks
        if recursive_splitting:
            chunks = self.text_splitter.split_text_recursive(full_text)
        else:
            chunks = self.text_splitter.split_text(full_text)
            
        # Prepare the result with document info and chunks
        result = {
            "text": full_text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "metadata": metadata
        }
        
        return result
        
    def process_directory(self, directory_path, include_metadata=False, recursive_splitting=True, recursive_dirs=True):
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path (str): Path to the directory containing documents
            include_metadata (bool): Whether to include document metadata
            recursive_splitting (bool): Whether to use recursive text splitting
            recursive_dirs (bool): Whether to recursively process subdirectories
            
        Returns:
            list: A list of dictionaries, each containing processed document information
        """
        results = []
        
        # Get all files in the directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, file_extension = os.path.splitext(file_path.lower())
                
                # Check if file type is supported
                if file_extension in self.loaders:
                    try:
                        # Process the document
                        processed_doc = self.process_document(
                            file_path, 
                            include_metadata=include_metadata,
                            recursive_splitting=recursive_splitting
                        )
                        results.append(processed_doc)
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        
            # If not recursive, break after the first directory
            if not recursive_dirs:
                break
                
        return results
        
    def register_loader(self, file_extension, loader):
        """
        Register a custom loader for a specific file extension.
        
        Args:
            file_extension (str): File extension (e.g., ".pdf", ".csv")
            loader: An instance of a loader class with a load_document method
        """
        # Normalize the extension with a leading period
        if not file_extension.startswith("."):
            file_extension = f".{file_extension}"
            
        self.loaders[file_extension.lower()] = loader 