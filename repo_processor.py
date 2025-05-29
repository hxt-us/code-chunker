import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    CSVLoader, 
    JSONLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import (
    PythonCodeTextSplitter,
    JavaScriptTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter
)

from code_chunker import CodeChunker
from semantic_chunker import SemanticChunker
from utils import get_file_info, detect_file_type, count_tokens

@dataclass
class FileMetadata:
    """Metadata for processed files"""
    file_path: str
    repo_name: str
    file_type: str
    language: str
    file_size: int
    lines_count: int
    last_modified: str
    relative_path: str
    directory_structure: List[str]
    imports_exports: List[str] = None
    functions_classes: List[str] = None

class RepositoryProcessor:
    """Main class for processing repositories with intelligent chunking"""
    
    def __init__(self, 
                 chunk_size: int = 2000,
                 chunk_overlap: int = 200,
                 token_limit: int = 4000,
                 encoding_name: str = "gpt-4"):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap  
        self.token_limit = token_limit
        self.encoding_name = encoding_name
        
        # Initialize chunkers
        self.semantic_chunker = SemanticChunker(encoding_name)
        
        # File type mappings
        self.code_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.kt': 'kotlin',
            '.swift': 'swift',
            '.scala': 'scala',
            '.r': 'r',
            '.sql': 'sql',
            '.sh': 'shell',
            '.ps1': 'powershell',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less'
        }
        
        self.document_extensions = {
            '.md': 'markdown',
            '.txt': 'text',
            '.csv': 'csv', 
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'doc',
            '.rtf': 'rtf'
        }
        
        # Initialize loggers
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all repositories in a directory"""
        directory_path = Path(directory_path)
        all_documents = []
        
        if not directory_path.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Find all repositories (directories with .git folder or containing code)
        repositories = self._find_repositories(directory_path)
        
        for repo_path in repositories:
            self.logger.info(f"Processing repository: {repo_path}")
            repo_documents = self._process_repository(repo_path)
            all_documents.extend(repo_documents)
        
        self.logger.info(f"Processed {len(repositories)} repositories, created {len(all_documents)} documents")
        return all_documents
    
    def _find_repositories(self, directory_path: Path) -> List[Path]:
        """Find all repository directories"""
        repositories = []
        
        # Check if current directory is a repo
        if (directory_path / '.git').exists():
            repositories.append(directory_path)
        else:
            # Look for subdirectories that are repos
            for item in directory_path.iterdir():
                if item.is_dir():
                    if (item / '.git').exists():
                        repositories.append(item)
                    elif self._contains_code_files(item):
                        repositories.append(item)
        
        return repositories
    
    def _contains_code_files(self, directory: Path, max_depth: int = 2) -> bool:
        """Check if directory contains code files (within max_depth)"""
        if max_depth <= 0:
            return False
            
        for item in directory.iterdir():
            if item.is_file():
                if item.suffix.lower() in self.code_extensions:
                    return True
            elif item.is_dir() and not item.name.startswith('.'):
                if self._contains_code_files(item, max_depth - 1):
                    return True
        return False
    
    def _process_repository(self, repo_path: Path) -> List[Document]:
        """Process a single repository"""
        repo_name = repo_path.name
        documents = []
        
        # Walk through all files in repository
        for file_path in self._get_processable_files(repo_path):
            try:
                file_documents = self._process_file(file_path, repo_name, repo_path)
                documents.extend(file_documents)
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        return documents
    
    def _get_processable_files(self, repo_path: Path) -> List[Path]:
        """Get all processable files from repository"""
        processable_files = []
        
        # Exclude patterns
        exclude_patterns = {
            '.git', '.gitignore', '.DS_Store', '__pycache__', 
            'node_modules', '.venv', 'venv', '.env', 
            'dist', 'build', '.pytest_cache', '.coverage',
            '.idea', '.vscode', '*.pyc', '*.pyo', '*.pyd'
        }
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                # Skip if in excluded directory
                if any(excluded in file_path.parts for excluded in exclude_patterns):
                    continue
                
                # Skip if excluded file
                if file_path.name in exclude_patterns:
                    continue
                
                # Check if file is processable
                if (file_path.suffix.lower() in self.code_extensions or 
                    file_path.suffix.lower() in self.document_extensions):
                    processable_files.append(file_path)
        
        return processable_files
    
    def _process_file(self, file_path: Path, repo_name: str, repo_path: Path) -> List[Document]:
        """Process a single file and return chunked documents"""
        
        # Create metadata
        metadata = self._create_file_metadata(file_path, repo_name, repo_path)
        
        # Read file content
        try:
            content = self._read_file_content(file_path)
        except Exception as e:
            self.logger.error(f"Cannot read file {file_path}: {str(e)}")
            return []
        
        # Determine chunking strategy based on file type
        file_extension = file_path.suffix.lower()
        
        if file_extension in self.code_extensions:
            return self._chunk_code_file(content, metadata, file_extension)
        elif file_extension in self.document_extensions:
            return self._chunk_document_file(content, metadata, file_extension)
        else:
            return self._chunk_generic_file(content, metadata)
    
    def _create_file_metadata(self, file_path: Path, repo_name: str, repo_path: Path) -> FileMetadata:
        """Create comprehensive metadata for a file"""
        
        # Calculate relative path
        relative_path = str(file_path.relative_to(repo_path))
        
        # Get directory structure
        directory_structure = list(file_path.relative_to(repo_path).parts[:-1])
        
        # Get file stats
        file_stat = file_path.stat()
        
        metadata = FileMetadata(
            file_path=str(file_path),
            repo_name=repo_name,
            file_type=detect_file_type(file_path.suffix),
            language=self.code_extensions.get(file_path.suffix.lower(), 'unknown'),
            file_size=file_stat.st_size,
            lines_count=0,  # Will be set after reading content
            last_modified=datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            relative_path=relative_path,
            directory_structure=directory_structure
        )
        
        return metadata
    
    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, read as binary and decode with errors='replace'
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    
    def _chunk_code_file(self, content: str, metadata: FileMetadata, file_extension: str) -> List[Document]:
        """Chunk code files using code-aware chunking"""
        
        # Update lines count
        metadata.lines_count = len(content.split('\n'))
        
        # Use existing CodeChunker for supported languages
        supported_extensions = ['py', 'js', 'jsx', 'ts', 'tsx', 'css', 'php', 'rb', 'go']
        ext_without_dot = file_extension.lstrip('.')
        
        if ext_without_dot in supported_extensions:
            try:
                code_chunker = CodeChunker(ext_without_dot, self.encoding_name)
                chunks = code_chunker.chunk(content, self.token_limit)
                
                # Extract code structure information
                if ext_without_dot in ['py', 'js', 'jsx', 'ts', 'tsx']:
                    metadata.imports_exports, metadata.functions_classes = self._extract_code_structure(
                        content, ext_without_dot
                    )
                
                documents = []
                for chunk_num, chunk_content in chunks.items():
                    doc_metadata = asdict(metadata)
                    doc_metadata.update({
                        'chunk_number': chunk_num,
                        'total_chunks': len(chunks),
                        'chunk_type': 'code',
                        'tokens': count_tokens(chunk_content, self.encoding_name)
                    })
                    
                    documents.append(Document(
                        page_content=chunk_content,
                        metadata=doc_metadata
                    ))
                
                return documents
                
            except Exception as e:
                self.logger.warning(f"CodeChunker failed for {metadata.file_path}: {str(e)}")
        
        # Fallback to language-specific text splitters
        return self._chunk_with_langchain_splitter(content, metadata, file_extension)
    
    def _chunk_with_langchain_splitter(self, content: str, metadata: FileMetadata, file_extension: str) -> List[Document]:
        """Use LangChain's language-specific splitters"""
        
        ext_without_dot = file_extension.lstrip('.')
        
        # Choose appropriate splitter
        if ext_without_dot == 'py':
            splitter = PythonCodeTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        elif ext_without_dot in ['js', 'jsx', 'ts', 'tsx']:
            splitter = JavaScriptTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        
        chunks = splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = asdict(metadata)
            doc_metadata.update({
                'chunk_number': i + 1,
                'total_chunks': len(chunks),
                'chunk_type': 'code',
                'tokens': count_tokens(chunk, self.encoding_name)
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def _chunk_document_file(self, content: str, metadata: FileMetadata, file_extension: str) -> List[Document]:
        """Chunk document files using semantic chunking"""
        
        metadata.lines_count = len(content.split('\n'))
        ext_without_dot = file_extension.lstrip('.')
        
        # Use semantic chunking for documents
        try:
            chunks = self.semantic_chunker.chunk_text(content, self.token_limit)
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = asdict(metadata)
                doc_metadata.update({
                    'chunk_number': i + 1,
                    'total_chunks': len(chunks),
                    'chunk_type': 'document',
                    'tokens': count_tokens(chunk, self.encoding_name)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            return documents
            
        except Exception as e:
            self.logger.warning(f"Semantic chunking failed for {metadata.file_path}: {str(e)}")
            return self._chunk_generic_file(content, metadata)
    
    def _chunk_generic_file(self, content: str, metadata: FileMetadata) -> List[Document]:
        """Generic chunking for unsupported file types"""
        
        metadata.lines_count = len(content.split('\n'))
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = asdict(metadata)
            doc_metadata.update({
                'chunk_number': i + 1,
                'total_chunks': len(chunks),
                'chunk_type': 'generic',
                'tokens': count_tokens(chunk, self.encoding_name)
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def _extract_code_structure(self, content: str, file_extension: str) -> Tuple[List[str], List[str]]:
        """Extract imports/exports and functions/classes from code"""
        
        imports_exports = []
        functions_classes = []
        
        lines = content.split('\n')
        
        if file_extension == 'py':
            for line in lines:
                line = line.strip()
                if line.startswith(('import ', 'from ')):
                    imports_exports.append(line)
                elif line.startswith(('def ', 'class ', 'async def ')):
                    functions_classes.append(line.split('(')[0].split(':')[0])
        
        elif file_extension in ['js', 'jsx', 'ts', 'tsx']:
            for line in lines:
                line = line.strip()
                if any(keyword in line for keyword in ['import ', 'export ', 'require(']):
                    imports_exports.append(line)
                elif any(keyword in line for keyword in ['function ', 'class ', 'const ', 'let ', 'var ']):
                    if any(symbol in line for symbol in ['=>', 'function', 'class']):
                        functions_classes.append(line.split('(')[0].split('{')[0].strip())
        
        return imports_exports[:10], functions_classes[:20]  # Limit to avoid too much metadata
    
    def save_processed_documents(self, documents: List[Document], output_path: str):
        """Save processed documents to JSON file"""
        
        serializable_docs = []
        for doc in documents:
            serializable_docs.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(documents)} documents to {output_path}")
    
    def get_processing_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        
        stats = {
            'total_documents': len(documents),
            'total_tokens': sum(doc.metadata.get('tokens', 0) for doc in documents),
            'repositories': {},
            'file_types': {},
            'languages': {},
            'chunk_types': {}
        }
        
        for doc in documents:
            metadata = doc.metadata
            
            # Repository stats
            repo_name = metadata.get('repo_name', 'unknown')
            if repo_name not in stats['repositories']:
                stats['repositories'][repo_name] = 0
            stats['repositories'][repo_name] += 1
            
            # File type stats
            file_type = metadata.get('file_type', 'unknown')
            if file_type not in stats['file_types']:
                stats['file_types'][file_type] = 0
            stats['file_types'][file_type] += 1
            
            # Language stats
            language = metadata.get('language', 'unknown')
            if language not in stats['languages']:
                stats['languages'][language] = 0
            stats['languages'][language] += 1
            
            # Chunk type stats
            chunk_type = metadata.get('chunk_type', 'unknown')
            if chunk_type not in stats['chunk_types']:
                stats['chunk_types'][chunk_type] = 0
            stats['chunk_types'][chunk_type] += 1
        
        return stats
