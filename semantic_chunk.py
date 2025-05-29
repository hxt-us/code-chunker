import re
import nltk
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from utils import count_tokens

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Falling back to rule-based chunking.")

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Using basic sentence splitting.")

@dataclass
class Sentence:
    text: str
    start_idx: int
    end_idx: int
    embedding: Optional[np.ndarray] = None

class SemanticChunker:
    """
    Semantic chunker that groups sentences based on semantic similarity
    and maintains document structure for better context preservation
    """
    
    def __init__(self, 
                 encoding_name: str = "gpt-4",
                 similarity_threshold: float = 0.7,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 2000):
        
        self.encoding_name = encoding_name
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_embeddings = True
            except Exception as e:
                logging.warning(f"Failed to load sentence transformer: {e}")
                self.use_embeddings = False
        else:
            self.use_embeddings = False
        
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, text: str, token_limit: int = 4000) -> List[str]:
        """
        Main method to chunk text semantically
        """
        if not text.strip():
            return []
        
        # Detect if text is structured (markdown, code comments, etc.)
        if self._is_structured_text(text):
            return self._chunk_structured_text(text, token_limit)
        else:
            return self._chunk_plain_text(text, token_limit)
    
    def _is_structured_text(self, text: str) -> bool:
        """Detect if text has markdown or other structure"""
        
        # Check for markdown headers
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return True
        
        # Check for bullet points or numbered lists
        if re.search(r'^\s*[-*+]\s+', text, re.MULTILINE):
            return True
        if re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            return True
        
        # Check for code blocks
        if '```' in text or re.search(r'^    \w', text, re.MULTILINE):
            return True
        
        # Check for consistent indentation (like documentation)
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('  ') or line.startswith('    '))
        if indented_lines > len(lines) * 0.3:  # More than 30% indented
            return True
        
        return False
    
    def _chunk_structured_text(self, text: str, token_limit: int) -> List[str]:
        """Chunk structured text (markdown, documentation, etc.)"""
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # Split by major sections (headers, code blocks, etc.)
        sections = self._split_into_sections(text)
        
        for section in sections:
            section_tokens = count_tokens(section, self.encoding_name)
            
            # If section is too large, split it further
            if section_tokens > token_limit:
                sub_chunks = self._chunk_large_section(section, token_limit)
                
                # Add current chunk if exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                chunks.extend(sub_chunks)
                
            # If adding this section would exceed limit
            elif current_tokens + section_tokens > token_limit:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                current_chunk = section
                current_tokens = section_tokens
                
            else:
                current_chunk += "\n\n" + section if current_chunk else section
                current_tokens += section_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > self.min_chunk_size]
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections"""
        
        sections = []
        
        # Split by markdown headers first
        header_pattern = r'^(#{1,6}\s+.*?)(?=\n#{1,6}\s+|\n*$)'
        header_matches = list(re.finditer(header_pattern, text, re.MULTILINE | re.DOTALL))
        
        if header_matches:
            last_end = 0
            
            for match in header_matches:
                # Add content before first header
                if match.start() > last_end:
                    pre_content = text[last_end:match.start()].strip()
                    if pre_content:
                        sections.append(pre_content)
                
                # Add header section
                sections.append(match.group(1).strip())
                last_end = match.end()
            
            # Add remaining content
            if last_end < len(text):
                remaining = text[last_end:].strip()
                if remaining:
                    sections.append(remaining)
        
        else:
            # Split by double newlines for paragraph-based content
            paragraphs = re.split(r'\n\s*\n', text)
            sections = [p.strip() for p in paragraphs if p.strip()]
        
        return sections
    
    def _chunk_large_section(self, section: str, token_limit: int) -> List[str]:
        """Chunk a large section that exceeds token limit"""
        
        # Try to split by sentences first
        if self.use_embeddings:
            return self._semantic_split_section(section, token_limit)
        else:
            return self._rule_based_split_section(section, token_limit)
    
    def _semantic_split_section(self, section: str, token_limit: int) -> List[str]:
        """Split section using semantic similarity"""
        
        sentences = self._split_into_sentences(section)
        if len(sentences) < 2:
            return [section]
        
        # Get embeddings for all sentences
        sentence_texts = [s.text for s in sentences]
        embeddings = self.sentence_model.encode(sentence_texts)
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            sentence.embedding = embedding
        
        # Group sentences semantically
        chunks = []
        current_group = [sentences[0]]
        current_tokens = count_tokens(sentences[0].text, self.encoding_name)
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = count_tokens(sentence.text, self.encoding_name)
            
            # Calculate similarity with current group
            if current_group:
                # Use average embedding of current group
                group_embeddings = np.array([s.embedding for s in current_group])
                group_avg = np.mean(group_embeddings, axis=0)
                similarity = cosine_similarity([sentence.embedding], [group_avg])[0][0]
                
                # Decide whether to add to current group or start new one
                if (similarity >= self.similarity_threshold and 
                    current_tokens + sentence_tokens <= token_limit):
                    current_group.append(sentence)
                    current_tokens += sentence_tokens
                else:
                    # Start new group
                    if current_group:
                        chunk_text = ' '.join(s.text for s in current_group)
                        chunks.append(chunk_text)
                    
                    current_group = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_group = [sentence]
                current_tokens = sentence_tokens
        
        # Add final group
        if current_group:
            chunk_text = ' '.join(s.text for s in current_group)
            chunks.append(chunk_text)
        
        return chunks
    
    def _rule_based_split_section(self, section: str, token_limit: int) -> List[str]:
        """Split section using rule-based approach"""
        
        sentences = self._split_into_sentences(section)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence.text, self.encoding_name)
            
            if current_tokens + sentence_tokens > token_limit:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence.text
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence.text if current_chunk else sentence.text
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_plain_text(self, text: str, token_limit: int) -> List[str]:
        """Chunk plain text without structure"""
        
        if self.use_embeddings:
            return self._semantic_chunk_plain_text(text, token_limit)
        else:
            return self._simple_chunk_plain_text(text, token_limit)
    
    def _semantic_chunk_plain_text(self, text: str, token_limit: int) -> List[str]:
        """Semantically chunk plain text"""
        
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return [text]
        
        # Get embeddings
        sentence_texts = [s.text for s in sentences]
        embeddings = self.sentence_model.encode(sentence_texts)
        
        # Find semantic boundaries
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Find low similarity points as potential boundaries
        threshold = np.percentile(similarities, 25)  # Bottom 25% as boundaries
        boundaries = [0]
        
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1)
        
        boundaries.append(len(sentences))
        
        # Create chunks from boundaries
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(s.text for s in chunk_sentences)
            chunk_tokens = count_tokens(chunk_text, self.encoding_name)
            
            # If chunk is too large, split it further
            if chunk_tokens > token_limit:
                sub_chunks = self._split_large_chunk(chunk_sentences, token_limit)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return [chunk for chunk in chunks if len(chunk.strip()) > self.min_chunk_size]
    
    def _split_large_chunk(self, sentences: List[Sentence], token_limit: int) -> List[str]:
        """Split a large chunk into smaller ones"""
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence.text, self.encoding_name)
            
            if current_tokens + sentence_tokens > token_limit:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence.text
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence.text if current_chunk else sentence.text
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _simple_chunk_plain_text(self, text: str, token_limit: int) -> List[str]:
        """Simple chunking for plain text"""
        
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence.text, self.encoding_name)
            
            if current_tokens + sentence_tokens > token_limit:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence.text
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence.text if current_chunk else sentence.text
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > self.min_chunk_size]
    
    def _split_into_sentences(self, text: str) -> List[Sentence]:
        """Split text into sentences with position tracking"""
        
        sentences = []
        
        if NLTK_AVAILABLE:
            try:
                sentence_spans = list(nltk.sent_tokenize(text))
                current_pos = 0
                
                for sent_text in sentence_spans:
                    # Find the sentence in the original text
                    start_idx = text.find(sent_text, current_pos)
                    if start_idx != -1:
                        end_idx = start_idx + len(sent_text)
                        sentences.append(Sentence(
                            text=sent_text.strip(),
                            start_idx=start_idx,
                            end_idx=end_idx
                        ))
                        current_pos = end_idx
                    else:
                        # Fallback: just use the sentence text
                        sentences.append(Sentence(
                            text=sent_text.strip(),
                            start_idx=current_pos,
                            end_idx=current_pos + len(sent_text)
                        ))
                        current_pos += len(sent_text)
                
            except Exception as e:
                self.logger.warning(f"NLTK sentence tokenization failed: {e}")
                sentences = self._simple_sentence_split(text)
        else:
            sentences = self._simple_sentence_split(text)
        
        return [s for s in sentences if s.text.strip()]
    
    def _simple_sentence_split(self, text: str) -> List[Sentence]:
        """Simple sentence splitting using regex"""
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentence_texts = re.split(sentence_pattern, text)
        
        sentences = []
        current_pos = 0
        
        for sent_text in sentence_texts:
            sent_text = sent_text.strip()
            if sent_text:
                start_idx = text.find(sent_text, current_pos)
                if start_idx != -1:
                    end_idx = start_idx + len(sent_text)
                else:
                    start_idx = current_pos
                    end_idx = current_pos + len(sent_text)
                
                sentences.append(Sentence(
                    text=sent_text,
                    start_idx=start_idx,
                    end_idx=end_idx
                ))
                current_pos = end_idx
        
        return sentences
