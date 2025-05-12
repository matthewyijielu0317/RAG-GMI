import re

class TextSplitter:
    """
    Split text into chunks of roughly equal size.
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, separator="\n"):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size (int): Target size of chunks (in characters)
            chunk_overlap (int): Number of characters to overlap between chunks
            separator (str): Default separator for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
    def split_text(self, text):
        """
        Split text into chunks of roughly equal size.
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            list: List of text chunks
        """
        chunks = []
        
        # Handle empty or small text
        if not text:
            return chunks
        
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to split on paragraph boundaries first
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk_size,
            # save the current chunk and start a new one
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                chunks.append(current_chunk)
                # Add overlap from previous chunk
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:]
                else:
                    current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            
            # If this single paragraph is too large, split it by sentences
            if len(current_chunk) > self.chunk_size:
                # Split by common sentence separators
                sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                current_chunk = ""
                
                for sentence in sentences:
                    if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                        chunks.append(current_chunk)
                        # Add overlap from previous chunk
                        if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                            current_chunk = current_chunk[-self.chunk_overlap:]
                        else:
                            current_chunk = ""
                    
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        # If we still have no chunks (maybe no newlines or sentence breaks),
        # fall back to simple chunking by size
        if not chunks:
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                end = min(i + self.chunk_size, len(text))
                chunks.append(text[i:end])
        
        return chunks
        
    def split_text_recursive(self, text):
        """
        Same as split_text but with a different name for backwards compatibility.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of chunks
        """
        return self.split_text(text) 