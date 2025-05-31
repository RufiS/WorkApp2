# Context processing utilities
import re
from typing import List, Dict, Any, Optional


def clean_context(text: str) -> str:
    """
    Clean context text for better readability

    Args:
        text: Raw context text

    Returns:
        Cleaned context text
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix hyphenation at line breaks
    text = re.sub(r"(\w)-\s*(\w)", r"\1\2", text)

    # Remove table of contents
    text = re.sub(r"Table of Contents.*?(?:\n|$)", "", text, flags=re.I | re.DOTALL)

    # Add paragraph breaks at appropriate places
    text = re.sub(r"\. ([A-Z])", r".\n\n\1", text)

    # Clean up any remaining issues
    text = text.strip()

    return text


def extract_hyperlinks(text: str) -> List[Dict[str, str]]:
    """
    Extract hyperlinks from text

    Args:
        text: Text to extract hyperlinks from

    Returns:
        List of dictionaries with 'text' and 'url' keys
    """
    # Pattern to match markdown-style links: [text](url)
    markdown_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    markdown_links = re.findall(markdown_pattern, text)

    # Pattern to match HTML-style links: <a href="url">text</a>
    html_pattern = r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>'
    html_links = re.findall(html_pattern, text)

    # Pattern to match raw URLs
    url_pattern = r"(?<!\()https?://[\w\d./?=&%-]+"
    raw_urls = re.findall(url_pattern, text)

    # Combine all links
    links = []

    # Add markdown links
    for link_text, url in markdown_links:
        links.append({"text": link_text, "url": url})

    # Add HTML links (note the order is reversed in the regex match)
    for url, link_text in html_links:
        links.append({"text": link_text, "url": url})

    # Add raw URLs
    for url in raw_urls:
        # Skip URLs that are already part of a markdown or HTML link
        if any(url == link["url"] for link in links):
            continue
        links.append({"text": url, "url": url})

    return links


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from text

    Args:
        text: Text to extract code blocks from

    Returns:
        List of dictionaries with 'language' and 'code' keys
    """
    # Pattern to match code blocks: ```language\ncode\n```
    pattern = r"```([a-zA-Z0-9]*)?\n([\s\S]*?)```"
    matches = re.findall(pattern, text)

    code_blocks = []
    for language, code in matches:
        code_blocks.append({"language": language.strip() or "text", "code": code.strip()})

    return code_blocks


def extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text

    Args:
        text: Text to extract bullet points from

    Returns:
        List of bullet points
    """
    # Pattern to match bullet points: - text or * text
    pattern = r"^[\s]*[-*]\s+(.+)$"
    matches = re.findall(pattern, text, re.MULTILINE)

    return [match.strip() for match in matches]


def extract_numbered_points(text: str) -> List[str]:
    """
    Extract numbered points from text

    Args:
        text: Text to extract numbered points from

    Returns:
        List of numbered points
    """
    # Pattern to match numbered points: 1. text or 1) text
    pattern = r"^[\s]*\d+[.)]\s+(.+)$"
    matches = re.findall(pattern, text, re.MULTILINE)

    return [match.strip() for match in matches]
