"""UI styling constants and helper functions"""

# Styling constants
WRAPPER_DIV_STYLE = "word-wrap: break-word; overflow-wrap: break-word; white-space: normal;"
UNCERTAIN_SECTION_STYLE = """
    background-color: #ffe0e0; 
    padding: 10px; 
    border-radius: 5px; 
    border: 1px dashed #ff8080; 
    margin-bottom: 10px;
"""
TERM_DISTINCTION_STYLE = """
    background-color: #e0f0ff;
    padding: 10px;
    border-radius: 5px;
    border: 1px dashed #80a0ff;
    margin-bottom: 10px;
    color: #cc0000;
"""

def get_html_wrapper(content: str, style: str) -> str:
    """Wrap content in a styled div
    
    Args:
        content: The HTML content to wrap
        style: CSS styles to apply
        
    Returns:
        HTML string with wrapped content
    """
    return f'<div style="{style}">{content}</div>'
