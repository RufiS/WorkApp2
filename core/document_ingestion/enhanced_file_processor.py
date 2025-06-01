"""
Enhanced File Processor - Fixes chunking issues and content filtering

Addresses:
1. Micro-chunking problem (2,477 tiny chunks → 200-300 meaningful chunks)
2. Table of contents footer noise filtering
3. Intelligent content cleaning for better LLM responses
4. Preserves legitimate TOC headers while removing footer pollution
"""

import os
import re
import time
import logging
import tempfile
from typing import List, Dict, Any, Tuple, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

from core.config import performance_config
from utils.loaders.pdf_hyperlink_loader import PDFHyperlinkLoader
from utils.common.error_handler import CommonErrorHandler, with_error_context
from utils.logging.error_logging import log_error, log_warning
from utils.error_handling.enhanced_decorators import with_advanced_retry, with_timing

logger = logging.getLogger(__name__)


class EnhancedFileProcessor:
    """Enhanced file processor with chunking fixes and content filtering"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the enhanced file processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Verify and adjust parameters
        self.chunk_size, self.chunk_overlap = self._verify_chunk_parameters()

        # Initialize content filters
        self._init_content_filters()

        logger.info(f"Enhanced file processor initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def _init_content_filters(self):
        """Initialize content filtering patterns"""
        # Table of contents footer patterns (to remove)
        self.toc_footer_patterns = [
            r'^table of contents\s*$',  # Simple "table of contents" lines
            r'^table\s+of\s+contents\s*$',  # Spaced version
            r'^toc\s*$',  # Abbreviated version
            r'^contents\s*$',  # Just "contents"
            r'^\s*\.\.\.\s*table\s+of\s+contents\s*\.\.\.\s*$',  # Decorated versions
            r'^\s*-+\s*table\s+of\s+contents\s*-+\s*$',  # Dashed versions
            r'^\s*table\s+of\s+contents\s+page\s+\d+\s*$',  # With page numbers
        ]

        # Legitimate TOC header patterns (to preserve)
        self.toc_header_patterns = [
            r'^table\s+of\s+contents\s*\n',  # Headers introducing actual TOC
            r'^contents\s*\n',  # Simple contents header
            r'^\s*table\s+of\s+contents\s*\n\s*[A-Z]',  # TOC followed by content
            r'^table\s+of\s+contents\s*\n.*\.\.\.\.\d+',  # TOC with dotted page numbers
        ]

        # Other noise patterns to filter
        self.noise_patterns = [
            r'^\s*page\s+\d+\s*$',  # Standalone page numbers
            r'^\s*\d+\s*$',  # Standalone numbers
            r'^\s*\.\s*$',  # Standalone dots
            r'^\s*-+\s*$',  # Lines of dashes
            r'^\s*=+\s*$',  # Lines of equals
            r'^\s*_+\s*$',  # Lines of underscores
            r'^\s*\*+\s*$',  # Lines of asterisks
        ]

        # Compile patterns for efficiency
        self.compiled_toc_footer = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                                   for pattern in self.toc_footer_patterns]
        self.compiled_toc_header = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                                   for pattern in self.toc_header_patterns]
        self.compiled_noise = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                              for pattern in self.noise_patterns]

    def _verify_chunk_parameters(self) -> Tuple[int, int]:
        """
        Verify and potentially adjust chunk size and overlap parameters

        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap

        # Ensure chunk size is reasonable for meaningful content
        if chunk_size < 500:
            logger.warning(f"Chunk size {chunk_size} too small for meaningful content, adjusting to 800")
            chunk_size = 800
        elif chunk_size > 2000:
            logger.warning(f"Chunk size {chunk_size} too large, adjusting to 1500")
            chunk_size = 1500

        # Ensure chunk overlap maintains context
        if chunk_overlap < 100:
            logger.warning(f"Chunk overlap {chunk_overlap} too small, adjusting to 150")
            chunk_overlap = 150
        elif chunk_overlap >= chunk_size // 2:
            overlap_target = chunk_size // 4
            logger.warning(f"Chunk overlap {chunk_overlap} too large, adjusting to {overlap_target}")
            chunk_overlap = overlap_target

        return chunk_size, chunk_overlap

    def filter_table_of_contents_noise(self, text: str) -> str:
        """
        Filter out table of contents footer noise while preserving legitimate headers

        Args:
            text: Raw text content

        Returns:
            Filtered text with TOC noise removed
        """
        lines = text.split('\n')
        filtered_lines = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Skip empty lines
            if not line_stripped:
                filtered_lines.append(line)
                continue

            # Check if this is a legitimate TOC header (preserve it)
            is_toc_header = False
            for pattern in self.compiled_toc_header:
                if pattern.search(line):
                    # Check context - is this introducing an actual table of contents?
                    context_lines = lines[i:i+5]  # Look ahead 5 lines
                    context = '\n'.join(context_lines)

                    # If followed by content with page numbers or structure, it's legitimate
                    if re.search(r'\.\.\.\.\d+|chapter\s+\d+|\d+\.\d+', context, re.IGNORECASE):
                        is_toc_header = True
                        break

            if is_toc_header:
                filtered_lines.append(line)
                continue

            # Check if this is footer TOC noise (remove it)
            is_toc_footer = False
            for pattern in self.compiled_toc_footer:
                if pattern.match(line_stripped):
                    is_toc_footer = True
                    logger.debug(f"Filtering TOC footer: '{line_stripped}'")
                    break

            if is_toc_footer:
                continue  # Skip this line

            # Check for other noise patterns
            is_noise = False
            for pattern in self.compiled_noise:
                if pattern.match(line_stripped):
                    is_noise = True
                    logger.debug(f"Filtering noise: '{line_stripped}'")
                    break

            if is_noise:
                continue  # Skip this line

            # Keep this line
            filtered_lines.append(line)

        filtered_text = '\n'.join(filtered_lines)

        # Log filtering results
        original_lines = len(lines)
        filtered_lines_count = len(filtered_lines)
        removed_lines = original_lines - filtered_lines_count

        if removed_lines > 0:
            logger.info(f"Content filtering: removed {removed_lines} noise lines from {original_lines} total lines")

        return filtered_text

    def enhance_text_extraction(self, text: str) -> str:
        """
        Enhance text extraction to prevent micro-chunking

        Args:
            text: Raw extracted text

        Returns:
            Enhanced text with better structure
        """
        # First, filter out noise
        text = self.filter_table_of_contents_noise(text)

        # Fix common PDF extraction issues that cause micro-chunking
        text = self._fix_pdf_extraction_issues(text)

        # Ensure minimum content density
        text = self._ensure_content_density(text)

        return text

    def _fix_pdf_extraction_issues(self, text: str) -> str:
        """Fix common PDF extraction issues that cause micro-chunking"""
        # Remove excessive whitespace that can cause splits
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double

        # Fix bullet points that get split inappropriately
        text = re.sub(r'\n\s*-\s*(?=[A-Z])', '\n- ', text)  # Normalize bullet points
        text = re.sub(r'\n\s*•\s*(?=[A-Z])', '\n• ', text)  # Normalize bullets

        # Keep related content together
        text = re.sub(r'\n(?=\s*[a-z])', ' ', text)  # Join lowercase continuations

        # Fix numbered lists that get broken
        text = re.sub(r'\n\s*(\d+\.)\s*(?=[A-Z])', r'\n\1 ', text)

        # Remove standalone punctuation that creates micro-chunks
        text = re.sub(r'\n\s*[.,;:]\s*\n', '\n', text)

        return text

    def _ensure_content_density(self, text: str) -> str:
        """Ensure text has sufficient content density for meaningful chunks"""
        lines = text.split('\n')
        dense_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Skip lines that are too short to be meaningful
            if len(line_stripped) < 10:
                # Only keep if it's part of a larger structure
                if line_stripped and not re.match(r'^[^\w]*$', line_stripped):
                    dense_lines.append(line)
                continue

            dense_lines.append(line)

        return '\n'.join(dense_lines)

    @with_error_context("file loader selection")
    def get_file_loader(self, file_path: str):
        """
        Get the appropriate loader for a file based on its extension

        Args:
            file_path: Path to the file

        Returns:
            A document loader instance
        """
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError(f"Not a file: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        # Support PDF, TXT, and DOCX files
        if file_ext == ".txt":
            return TextLoader(file_path)
        elif file_ext == ".pdf":
            if performance_config.extract_pdf_hyperlinks:
                return PDFHyperlinkLoader(file_path)
            else:
                return PyPDFLoader(file_path)
        elif file_ext in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {file_ext}. Only PDF, TXT, and DOCX files are supported."
            )

    @with_timing(threshold=1.0)
    @with_advanced_retry(max_attempts=3, backoff_factor=2)
    def process_file_upload(self, file) -> List[Dict[str, Any]]:
        """
        Process a file object (from streamlit file uploader)

        Args:
            file: File object from streamlit file uploader

        Returns:
            List of document chunks with metadata
        """
        temp_file_path = None
        try:
            # Validate file
            if not file or not hasattr(file, "getvalue") or len(file.getvalue()) == 0:
                error_msg = f"Empty or invalid file: {getattr(file, 'name', 'unknown')}"
                log_error(error_msg, include_traceback=False)
                raise ValueError(error_msg)

            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(file.name)[1]
            ) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name

            # Process the temporary file
            chunks = self.load_and_chunk_document(temp_file_path)

            # Check if chunks are empty and log it
            if not chunks:
                error_msg = f"No content extracted from file {file.name}. File may be empty, corrupted, or in an unsupported format."
                logger.warning(error_msg)
                log_warning(error_msg, include_traceback=False)

            return chunks

        except Exception as e:
            error_msg = f"Error processing file {getattr(file, 'name', 'unknown')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            log_error(error_msg, include_traceback=True)
            raise RuntimeError(error_msg) from e

        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file_path}: {str(e)}")

    @with_timing(threshold=1.0)
    @with_error_context("document loading and chunking")
    def load_and_chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document and split it into chunks with enhanced processing

        Args:
            file_path: Path to the document file

        Returns:
            List of document chunks with metadata
        """
        # Validate file
        self._validate_file(file_path)

        try:
            # Get appropriate loader
            loader = self.get_file_loader(file_path)

            # Load document
            documents = self._load_document(loader, file_path)

            # Check if documents were loaded
            if not documents:
                warning_msg = f"No content loaded from {os.path.basename(file_path)}"
                logger.warning(warning_msg)
                log_warning(warning_msg)
                return []

            # Enhanced text processing
            enhanced_documents = self._enhance_document_content(documents)

            # Create and configure text splitter with enhanced settings
            text_splitter = self._create_enhanced_text_splitter()

            # Log chunking parameters
            logger.info(
                f"Enhanced chunking for {os.path.basename(file_path)} with size={self.chunk_size}, overlap={self.chunk_overlap}"
            )

            # Split documents into chunks
            chunks = self._split_documents_enhanced(text_splitter, enhanced_documents, file_path)

            # Format chunks with metadata
            formatted_chunks = self._format_chunks_enhanced(chunks, file_path)

            # Validate chunk quality
            self._validate_chunk_quality(formatted_chunks, file_path)

            logger.info(f"Enhanced processing complete for {file_path}: {len(formatted_chunks)} chunks")
            return formatted_chunks

        except Exception as e:
            error_msg = f"Error in enhanced processing for {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            log_error(error_msg)
            raise

    def _enhance_document_content(self, documents):
        """Apply content enhancement to documents"""
        enhanced_docs = []

        for doc in documents:
            # Apply text enhancement
            enhanced_content = self.enhance_text_extraction(doc.page_content)

            # Create new document with enhanced content
            from langchain.schema import Document
            enhanced_doc = Document(
                page_content=enhanced_content,
                metadata=doc.metadata
            )
            enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def _create_enhanced_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create enhanced text splitter with optimized settings"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Enhanced separators
            keep_separator=True,  # Keep context
            length_function=len,
            is_separator_regex=False,
        )

    def _split_documents_enhanced(self, text_splitter, documents, file_path: str):
        """Enhanced document splitting with validation"""
        try:
            chunks = text_splitter.split_documents(documents)

            # Log detailed chunking info
            total_content = sum(len(doc.page_content) for doc in documents)
            avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0

            logger.info(f"Enhanced splitting results:")
            logger.info(f"  - Original content: {total_content} characters")
            logger.info(f"  - Generated chunks: {len(chunks)}")
            logger.info(f"  - Average chunk size: {avg_chunk_size:.1f} characters")

            return chunks
        except Exception as e:
            error_msg = f"Error in enhanced chunking for {file_path}: {str(e)}"
            log_error(error_msg)
            raise RuntimeError(f"Failed to chunk document: {str(e)}") from e

    def _format_chunks_enhanced(self, chunks, file_path: str) -> List[Dict[str, Any]]:
        """Enhanced chunk formatting with quality checks"""
        formatted_chunks = []

        for i, chunk in enumerate(chunks):
            # Enhanced quality filtering
            if not self._is_chunk_valid(chunk.page_content):
                continue

            # Create enhanced chunk metadata
            chunk_data = {
                "id": f"{os.path.basename(file_path)}-{i}",
                "text": chunk.page_content,
                "metadata": {
                    "source": file_path,
                    "page": chunk.metadata.get("page", None),
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    "creation_time": time.time(),
                    "chunk_params": {
                        "size": self.chunk_size,
                        "overlap": self.chunk_overlap,
                    },
                    "processing": {
                        "enhanced": True,
                        "toc_filtered": True,
                        "content_density_checked": True,
                    },
                },
            }

            formatted_chunks.append(chunk_data)

        return formatted_chunks

    def _is_chunk_valid(self, content: str) -> bool:
        """Enhanced chunk validation"""
        # Must have minimum meaningful content
        if len(content.strip()) < 50:
            return False

        # Must contain some alphabetic characters (not just punctuation/numbers)
        if not re.search(r'[a-zA-Z]', content):
            return False

        # Must not be just noise patterns
        content_stripped = content.strip()
        for pattern in self.compiled_noise:
            if pattern.match(content_stripped):
                return False

        # Must not be just TOC footer
        for pattern in self.compiled_toc_footer:
            if pattern.match(content_stripped):
                return False

        return True

    def _validate_chunk_quality(self, chunks: List[Dict[str, Any]], file_path: str):
        """Validate overall chunk quality and log warnings"""
        if not chunks:
            logger.error(f"No valid chunks generated for {file_path}")
            return

        # Analyze chunk size distribution
        sizes = [len(chunk["text"]) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)

        # Log quality metrics
        logger.info(f"Chunk quality analysis for {os.path.basename(file_path)}:")
        logger.info(f"  - Total chunks: {len(chunks)}")
        logger.info(f"  - Size range: {min_size}-{max_size} chars")
        logger.info(f"  - Average size: {avg_size:.1f} chars")

        # Quality warnings
        if len(chunks) > 500:
            logger.warning(f"High chunk count ({len(chunks)}) may indicate micro-chunking")

        if avg_size < 200:
            logger.warning(f"Low average chunk size ({avg_size:.1f}) may indicate content fragmentation")

        # Count chunks with expected content types
        text_message_chunks = sum(1 for chunk in chunks
                                 if re.search(r'text\s+message|sms|texting', chunk["text"], re.IGNORECASE))

        if text_message_chunks > 0:
            logger.info(f"Found {text_message_chunks} chunks with text messaging content")
        else:
            logger.warning("No chunks found with text messaging content - may need content review")

    def _validate_file(self, file_path: str) -> None:
        """Validate file existence and accessibility"""
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise FileNotFoundError(error_msg)

        if not os.access(file_path, os.R_OK):
            error_msg = f"Permission denied: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise PermissionError(error_msg)

        if os.path.getsize(file_path) == 0:
            error_msg = f"Empty file: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise ValueError(error_msg)

    def _load_document(self, loader, file_path: str):
        """Load document using the provided loader"""
        try:
            documents = loader.load()
            return documents
        except Exception as e:
            error_msg = f"Error loading document {file_path}: {str(e)}"
            log_error(error_msg)
            raise RuntimeError(f"Failed to load document: {str(e)}") from e

    def get_content_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed content statistics for analysis"""
        if not chunks:
            return {"error": "No chunks to analyze"}

        # Size analysis
        sizes = [len(chunk["text"]) for chunk in chunks]

        # Content analysis
        text_message_content = sum(1 for chunk in chunks
                                  if re.search(r'text\s+message|sms|texting|ringcentral',
                                             chunk["text"], re.IGNORECASE))

        toc_content = sum(1 for chunk in chunks
                         if re.search(r'table\s+of\s+contents', chunk["text"], re.IGNORECASE))

        return {
            "total_chunks": len(chunks),
            "size_stats": {
                "min": min(sizes),
                "max": max(sizes),
                "average": sum(sizes) / len(sizes),
                "median": sorted(sizes)[len(sizes)//2],
            },
            "content_analysis": {
                "text_message_chunks": text_message_content,
                "toc_chunks": toc_content,
                "text_message_percentage": (text_message_content / len(chunks)) * 100,
            },
            "quality_indicators": {
                "appropriate_chunk_count": 100 <= len(chunks) <= 400,
                "good_average_size": 300 <= (sum(sizes) / len(sizes)) <= 1200,
                "has_target_content": text_message_content > 0,
            }
        }
