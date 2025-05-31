# PDF Hyperlink Loader
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

# Setup logging
logger = logging.getLogger(__name__)


class PDFHyperlinkLoader:
    """Loader that extracts text and hyperlinks from PDF files"""

    def __init__(self, file_path: str):
        """Initialize with file path

        Args:
            file_path: Path to the PDF file
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load PDF and extract text with hyperlinks

        Returns:
            List of Document objects with page content and metadata
        """
        documents = []

        try:
            # Open the PDF file
            pdf_document = fitz.open(self.file_path)

            # Process each page
            for page_num, page in enumerate(pdf_document):
                try:
                    # Get the page text
                    text = page.get_text("text")

                    # Get links from the page
                    links = page.get_links()

                    # Process and insert hyperlinks into the text
                    text_with_links = self._process_links(text, links, page)

                    # Create a Document object
                    doc = Document(
                        page_content=text_with_links,
                        metadata={
                            "source": self.file_path,
                            "page": page_num + 1,  # 1-indexed page numbers
                        },
                    )

                    documents.append(doc)
                except Exception as e:
                    # If hyperlink processing fails, fall back to just extracting text
                    logger.warning(
                        f"Error processing hyperlinks on page {page_num + 1}: {str(e)}. Falling back to text-only extraction."
                    )
                    text = page.get_text("text")
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": self.file_path,
                            "page": page_num + 1,
                        },
                    )
                    documents.append(doc)

            pdf_document.close()

        except Exception as e:
            logger.error(f"Error loading PDF {self.file_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error loading PDF {self.file_path}: {str(e)}")

        return documents

    def _process_links(self, text: str, links: List[Dict[str, Any]], page: Any) -> str:
        """Process links and insert them into the text

        Args:
            text: The page text
            links: List of link dictionaries
            page: The PDF page object

        Returns:
            Text with hyperlinks inserted
        """
        if not links:
            return text

        # Process each link
        for link in links:
            try:
                if "uri" in link and link["uri"]:
                    # Get the rectangle for this link
                    rect = None

                    # Try different ways to get the rectangle based on PyMuPDF version
                    if hasattr(page, "rect_for_link"):
                        # Older versions of PyMuPDF
                        rect = page.rect_for_link(link)
                    elif "from" in link:
                        # Newer versions of PyMuPDF
                        rect = fitz.Rect(link["from"])
                    elif "rect" in link:
                        # Some versions might have it directly as 'rect'
                        rect = fitz.Rect(link["rect"])
                    else:
                        # Skip links without position information
                        continue

                    # Get the text within this rectangle
                    link_text = ""
                    if hasattr(page, "get_textbox"):
                        # Older versions of PyMuPDF
                        link_text = page.get_textbox(rect)
                    else:
                        # Newer versions of PyMuPDF
                        link_text = page.get_text("text", clip=rect)

                    link_text = link_text.strip()

                    if link_text:
                        # Replace the link text with text that includes the URL
                        # Format: original_text [URL: actual_url]
                        replacement = f"{link_text} [URL: {link['uri']}]"

                        # Replace in the main text, being careful about exact matches
                        # We use a simple replacement strategy here
                        text = text.replace(link_text, replacement, 1)
            except Exception as e:
                # Log the error but continue processing other links
                logger.warning(f"Error processing link: {str(e)}")
                continue

        return text
