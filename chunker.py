from typing import List
import nltk
from nltk.tokenize import sent_tokenize

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
    def chunk_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                # Зберігаємо поточний чанк
                chunks.append(" ".join(current_chunk))
                
                # Зберігаємо останні речення для перекриття
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks 