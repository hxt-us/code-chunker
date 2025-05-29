from abc import ABC, abstractmethod
import re
import ast
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
import logging

from utils import count_tokens

# Import your existing CodeParser
try:
    from CodeParser import CodeParser
except ImportError:
    CodeParser = None
    logging.warning("CodeParser not available. Using fallback methods.")

@dataclass 
class CodeBlock:
    """Represents a code block with metadata"""
    content: str
    start_line: int
    end_line: int
    block_type: str  # function, class, import, etc.
    name: str = ""
    parent: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class Chunker(ABC):
    def __init__(self, encoding_name="gpt-4"):
        self.encoding_name = encoding_name

    @abstractmethod
    def chunk(self, content, token_limit):
        pass

    @abstractmethod
    def get_chunk(self, chunked_content, chunk_number):
        pass

    @staticmethod
    def print_chunks(chunks):
        for chunk_number, chunk_code in chunks.items():
            print(f"Chunk {chunk_number}:")
            print("=" * 40)
            print(chunk_code)
            print("=" * 40)

    @staticmethod
    def consolidate_chunks_into_file(chunks):
        return "\n".join(chunks.values())

    @staticmethod
    def count_lines(consolidated_chunks):
        lines = consolidated_chunks.split("\n")
        return len(lines)


class CodeChunker(Chunker):
    """Enhanced code chunker with better structure preservation"""
    
    def __init__(self, file_extension, encoding_name="gpt-4"):
        super().__init__(encoding_name)
        self.file_extension = file_extension
        self.logger = logging.getLogger(__name__)

    def chunk(self, code: str, token_limit: int) -> Dict[int, str]:
        """Enhanced chunking that preserves code structure"""
        
        # Try to use existing CodeParser first
        if CodeParser and self.file_extension in ['py', 'js', 'jsx', 'ts', 'tsx', 'css', 'php', 'rb', 'go']:
            try:
                return self._chunk_with_tree_sitter(code, token_limit)
            except Exception as e:
                self.logger.warning(f"Tree-sitter chunking failed: {e}")
        
        # Fallback to AST-based chunking for Python
        if self.file_extension == 'py':
            try:
                return self._chunk_python_ast(code, token_limit)
            except Exception as e:
                self.logger.warning(f"AST chunking failed: {e}")
        
        # Fallback to regex-based chunking
        return self._chunk_with_regex(code, token_limit)
    
    def _chunk_with_tree_sitter(self, code: str, token_limit: int) -> Dict[int, str]:
        """Use existing CodeParser with tree-sitter"""
        
        code_parser = CodeParser(self.file_extension)
        chunks = {}
        current_chunk = ""
        token_count = 0
        lines = code.split("\n")
        i = 0
        chunk_number = 1
        start_line = 0
        
        # Get breakpoints from CodeParser
        breakpoints = sorted(code_parser.get_lines_for_points_of_interest(code, self.file_extension))
        comments = sorted(code_parser.get_lines_for_comments(code, self.file_extension))
        
        # Adjust breakpoints to include preceding comments
        adjusted_breakpoints = []
        for bp in breakpoints:
            current_line = bp - 1
            highest_comment_line = None
            
            while current_line in comments:
                highest_comment_line = current_line
                current_line -= 1

            if highest_comment_line is not None:
                adjusted_breakpoints.append(highest_comment_line)
            else:
                adjusted_breakpoints.append(bp)

        breakpoints = sorted(set(adjusted_breakpoints))

        # Enhanced chunking logic with better boundary detection
        while i < len(lines):
            line = lines[i]
            new_token_count = count_tokens(line, self.encoding_name)
            
            if token_count + new_token_count > token_limit:
                # Find the best breakpoint
                if i in breakpoints:
                    stop_line = i
                else:
                    # Find the closest breakpoint before current line
                    valid_breakpoints = [x for x in breakpoints if x < i]
                    stop_line = max(valid_breakpoints, default=start_line) if valid_breakpoints else start_line

                # Create chunk if we have content
                if stop_line > start_line:
                    current_chunk = "\n".join(lines[start_line:stop_line])
                    if current_chunk.strip():
                        chunks[chunk_number] = current_chunk
                        chunk_number += 1
                    
                    start_line = stop_line
                    token_count = 0
                    i = stop_line
                else:
                    # Force include current line to avoid infinite loop
                    token_count += new_token_count
                    i += 1
            else:
                token_count += new_token_count
                i += 1

        # Add remaining code
        if start_line < len(lines):
            current_chunk = "\n".join(lines[start_line:])
            if current_chunk.strip():
                chunks[chunk_number] = current_chunk

        return chunks
    
    def _chunk_python_ast(self, code: str, token_limit: int) -> Dict[int, str]:
        """Chunk Python code using AST analysis"""
        
        try:
            tree = ast.parse(code)
            code_blocks = self._extract_python_blocks(tree, code)
            return self._create_chunks_from_blocks(code_blocks, code, token_limit)
        except SyntaxError as e:
            self.logger.warning(f"Python AST parsing failed: {e}")
            return self._chunk_with_regex(code, token_limit)
    
    def _extract_python_blocks(self, tree: ast.AST, code: str) -> List[CodeBlock]:
        """Extract code blocks from Python AST"""
        
        blocks = []
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno - 1
                end_line = self._find_block_end(node, lines)
                
                blocks.append(CodeBlock(
                    content='\n'.join(lines[start_line:end_line + 1]),
                    start_line=start_line,
                    end_line=end_line,
                    block_type='function',
                    name=node.name,
                    dependencies=self._extract_function_dependencies(node)
                ))
                
            elif isinstance(node, ast.ClassDef):
                start_line = node.lineno - 1
                end_line = self._find_block_end(node, lines)
                
                blocks.append(CodeBlock(
                    content='\n'.join(lines[start_line:end_line + 1]),
                    start_line=start_line,
                    end_line=end_line,
                    block_type='class',
                    name=node.name
                ))
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno - 1
                end_line = node.end_lineno - 1 if node.end_lineno else start_line
                
                blocks.append(CodeBlock(
                    content='\n'.join(lines[start_line:end_line + 1]),
                    start_line=start_line,
                    end_line=end_line,
                    block_type='import',
                    name=self._extract_import_name(node)
                ))
        
        # Sort blocks by start line
        blocks.sort(key=lambda x: x.start_line)
        
        # Fill gaps with miscellaneous code blocks
        filled_blocks = self._fill_gaps(blocks, lines)
        
        return filled_blocks
    
    def _find_block_end(self, node: ast.AST, lines: List[str]) -> int:
        """Find the end line of an AST node"""
        
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - 1
        
        # Fallback: find by indentation
        start_line = node.lineno - 1
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Skip empty lines
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= base_indent:
                    return i - 1
        
        return len(lines) - 1
    
    def _extract_function_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract function dependencies from AST node"""
        
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.append(child.id)
            elif isinstance(child, ast.Attribute):
                dependencies.append(child.attr)
        
        return list(set(dependencies))[:10]  # Limit to avoid too many deps
    
    def _extract_import_name(self, node: Union[ast.Import, ast.ImportFrom]) -> str:
        """Extract import name from import node"""
        
        if isinstance(node, ast.Import):
            return ', '.join(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = ', '.join(alias.name for alias in node.names)
            return f"{module}: {names}"
        
        return ""
    
    def _fill_gaps(self, blocks: List[CodeBlock], lines: List[str]) -> List[CodeBlock]:
        """Fill gaps between blocks with miscellaneous code"""
        
        filled_blocks = []
        current_line = 0
        
        for block in blocks:
            # Add gap before block if exists
            if current_line < block.start_line:
                gap_content = '\n'.join(lines[current_line:block.start_line])
                if gap_content.strip():
                    filled_blocks.append(CodeBlock(
                        content=gap_content,
                        start_line=current_line,
                        end_line=block.start_line - 1,
                        block_type='misc'
                    ))
            
            filled_blocks.append(block)
            current_line = block.end_line + 1
        
        # Add remaining lines
        if current_line < len(lines):
            remaining_content = '\n'.join(lines[current_line:])
            if remaining_content.strip():
                filled_blocks.append(CodeBlock(
                    content=remaining_content,
                    start_line=current_line,
                    end_line=len(lines) - 1,
                    block_type='misc'
                ))
        
        return filled_blocks
    
    def _create_chunks_from_blocks(self, blocks: List[CodeBlock], code: str, token_limit: int) -> Dict[int, str]:
        """Create chunks from code blocks"""
        
        chunks = {}
        chunk_number = 1
        current_chunk_blocks = []
        current_tokens = 0
        
        for block in blocks:
            block_tokens = count_tokens(block.content, self.encoding_name)
            
            # If single block exceeds limit, split it
            if block_tokens > token_limit:
                # Save current chunk if exists
                if current_chunk_blocks:
                    chunk_content = '\n\n'.join(b.content for b in current_chunk_blocks)
                    chunks[chunk_number] = chunk_content
                    chunk_number += 1
                    current_chunk_blocks = []
                    current_tokens = 0
                
                # Split large block
                sub_chunks = self._split_large_block(block, token_limit)
                for sub_chunk in sub_chunks:
                    chunks[chunk_number] = sub_chunk
                    chunk_number += 1
                
            # If adding block would exceed limit
            elif current_tokens + block_tokens > token_limit:
                if current_chunk_blocks:
                    chunk_content = '\n\n'.join(b.content for b in current_chunk_blocks)
                    chunks[chunk_number] = chunk_content
                    chunk_number += 1
                
                current_chunk_blocks = [block]
                current_tokens = block_tokens
            
            else:
                current_chunk_blocks.append(block)
                current_tokens += block_tokens
        
        # Add final chunk
        if current_chunk_blocks:
            chunk_content = '\n\n'.join(b.content for b in current_chunk_blocks)
            chunks[chunk_number] = chunk_content
        
        return chunks
    
    def _split_large_block(self, block: CodeBlock, token_limit: int) -> List[str]:
        """Split a large code block"""
        
        lines = block.content.split('\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line, self.encoding_name)
            
            if current_tokens + line_tokens > token_limit:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
                current_tokens = line_tokens
            else:
                current_chunk += '\n' + line if current_chunk else line
                current_tokens += line_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_with_regex(self, code: str, token_limit: int) -> Dict[int, str]:
        """Fallback regex-based chunking"""
        
        # Define patterns for different languages
        patterns = self._get_code_patterns()
        
        chunks = {}
        chunk_number = 1
        lines = code.split('\n')
        current_chunk = ""
        current_tokens = 0
        i = 0
        
        # Find all function/class boundaries
        boundaries = set()
        for line_num, line in enumerate(lines):
            for pattern in patterns:
                if re.match(pattern, line.strip()):
                    boundaries.add(line_num)
                    break
        
        boundaries = sorted(boundaries)
        
        while i < len(lines):
            line = lines[i]
            line_tokens = count_tokens(line, self.encoding_name)
            
            if current_tokens + line_tokens > token_limit:
                # Find nearest boundary
                boundary = None
                for b in boundaries:
                    if b < i:
                        boundary = b
                    else:
                        break
                
                if boundary is not None and boundary > i - 20:  # Don't go too far back
                    # Create chunk up to boundary
                    chunk_content = '\n'.join(lines[i - len(current_chunk.split('\n')) + 1:boundary])
                    if chunk_content.strip():
                        chunks[chunk_number] = chunk_content.strip()
                        chunk_number += 1
                    
                    # Start new chunk from boundary
                    current_chunk = '\n'.join(lines[boundary:i + 1])
                    current_tokens = count_tokens(current_chunk, self.encoding_name)
                else:
                    # No good boundary, just split here
                    if current_chunk:
                        chunks[chunk_number] = current_chunk.strip()
                        chunk_number += 1
                    
                    current_chunk = line
                    current_tokens = line_tokens
            else:
                current_chunk += '\n' + line if current_chunk else line
                current_tokens += line_tokens
            
            i += 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks[chunk_number] = current_chunk.strip()
        
        return chunks
    
    def _get_code_patterns(self) -> List[str]:
        """Get regex patterns for code structure based on language"""
        
        if self.file_extension == 'py':
            return [
                r'^def\s+\w+',
                r'^class\s+\w+',
                r'^async\s+def\s+\w+',
                r'^@\w+',
                r'^from\s+\w+',
                r'^import\s+\w+'
            ]
        elif self.file_extension in ['js', 'jsx', 'ts', 'tsx']:
            return [
                r'^function\s+\w+',
                r'^const\s+\w+\s*=\s*\(',
                r'^let\s+\w+\s*=\s*\(',
                r'^var\s+\w+\s*=\s*\(',
                r'^class\s+\w+',
                r'^export\s+',
                r'^import\s+'
            ]
        elif self.file_extension == 'java':
            return [
                r'^public\s+class\s+\w+',
                r'^private\s+class\s+\w+',
                r'^public\s+\w+.*\(',
                r'^private\s+\w+.*\(',
                r'^protected\s+\w+.*\('
            ]
        else:
            return [
                r'^\w+.*\{',  # Generic function/block pattern
                r'^class\s+\w+',
                r'^function\s+\w+'
            ]

    def get_chunk(self, chunked_codebase, chunk_number):
        return chunked_codebase[chunk_number]
