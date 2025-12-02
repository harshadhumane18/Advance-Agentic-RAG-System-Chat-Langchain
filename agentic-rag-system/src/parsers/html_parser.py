"""HTML parsing utilities for document extraction."""

import re
from typing import Generator, Optional
from bs4 import BeautifulSoup, Doctype, NavigableString, Tag
from loguru import logger


class HTMLParser:
    """HTML parser for converting web content to clean text."""
    
    def __init__(self):
        """Initialize the HTML parser."""
        self.scrape_tags = ["nav", "footer", "aside", "script", "style", "header"]
    
    def parse_html(self, html_content: str, url: Optional[str] = None) -> str:
        """Parse HTML content and extract clean text.
        
        Args:
            html_content: Raw HTML content
            url: Source URL for logging
            
        Returns:
            Clean, structured text
        """
        try:
            soup = BeautifulSoup(html_content, "lxml")
            
            # Remove unwanted tags
            for tag in soup.find_all(self.scrape_tags):
                tag.decompose()
            
            # Extract text with structure preservation
            text = self._extract_structured_text(soup)
            
            # Clean up whitespace
            text = re.sub(r"\n\n+", "\n\n", text).strip()
            
            logger.info(f"Successfully parsed HTML content from {url or 'unknown source'}")
            return text
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            raise
    
    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """Extract structured text from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Structured text content
        """
        def get_text(tag: Tag) -> Generator[str, None, None]:
            for child in tag.children:
                if isinstance(child, Doctype):
                    continue
                
                if isinstance(child, NavigableString):
                    yield str(child)
                elif isinstance(child, Tag):
                    yield from self._process_tag(child)
        
        return "".join(get_text(soup))
    
    def _process_tag(self, tag: Tag) -> Generator[str, None, None]:
        """Process individual HTML tags and convert to markdown-like format.
        
        Args:
            tag: HTML tag to process
            
        Yields:
            Formatted text content
        """
        tag_name = tag.name.lower()
        
        if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag_name[1:])
            yield f"{'#' * level} {tag.get_text()}\n\n"
        
        elif tag_name == "a":
            href = tag.get("href", "")
            text = tag.get_text(strip=False)
            yield f"[{text}]({href})"
        
        elif tag_name == "img":
            alt = tag.get("alt", "")
            src = tag.get("src", "")
            yield f"![{alt}]({src})"
        
        elif tag_name in ["strong", "b"]:
            yield f"**{tag.get_text(strip=False)}**"
        
        elif tag_name in ["em", "i"]:
            yield f"_{tag.get_text(strip=False)}_"
        
        elif tag_name == "br":
            yield "\n"
        
        elif tag_name == "code":
            parent = tag.find_parent()
            if parent and parent.name == "pre":
                # Handle code blocks
                classes = parent.attrs.get("class", [])
                language = self._extract_language(classes)
                code_content = self._extract_code_content(tag)
                yield f"```{language}\n{code_content}\n```\n\n"
            else:
                # Inline code
                yield f"`{tag.get_text(strip=False)}`"
        
        elif tag_name == "p":
            for child in tag.children:
                if isinstance(child, NavigableString):
                    yield str(child)
                elif isinstance(child, Tag):
                    yield from self._process_tag(child)
            yield "\n\n"
        
        elif tag_name == "ul":
            for li in tag.find_all("li", recursive=False):
                yield "- "
                for child in li.children:
                    if isinstance(child, NavigableString):
                        yield str(child)
                    elif isinstance(child, Tag):
                        yield from self._process_tag(child)
                yield "\n\n"
        
        elif tag_name == "ol":
            for i, li in enumerate(tag.find_all("li", recursive=False)):
                yield f"{i + 1}. "
                for child in li.children:
                    if isinstance(child, NavigableString):
                        yield str(child)
                    elif isinstance(child, Tag):
                        yield from self._process_tag(child)
                yield "\n\n"
        
        elif tag_name == "table":
            yield from self._process_table(tag)
        
        else:
            # Process children recursively
            for child in tag.children:
                if isinstance(child, NavigableString):
                    yield str(child)
                elif isinstance(child, Tag):
                    yield from self._process_tag(child)
    
    def _extract_language(self, classes: list) -> str:
        """Extract programming language from CSS classes.
        
        Args:
            classes: List of CSS classes
            
        Returns:
            Programming language name
        """
        for class_name in classes:
            match = re.match(r"language-(\w+)", class_name)
            if match:
                return match.group(1)
        return ""
    
    def _extract_code_content(self, code_tag: Tag) -> str:
        """Extract code content from code tag.
        
        Args:
            code_tag: Code tag element
            
        Returns:
            Code content
        """
        # Handle syntax highlighted code
        spans = code_tag.find_all("span", class_="token-line")
        if spans:
            lines = []
            for span in spans:
                line_content = "".join(token.get_text() for token in span.find_all("span"))
                lines.append(line_content)
            return "\n".join(lines)
        else:
            return code_tag.get_text()
    
    def _process_table(self, table_tag: Tag) -> Generator[str, None, None]:
        """Process HTML table and convert to markdown format.
        
        Args:
            table_tag: Table tag element
            
        Yields:
            Markdown-formatted table
        """
        thead = table_tag.find("thead")
        if thead:
            headers = thead.find_all("th")
            if headers:
                yield "| " + " | ".join(header.get_text() for header in headers) + " |\n"
                yield "| " + " | ".join("----" for _ in headers) + " |\n"
        
        tbody = table_tag.find("tbody")
        if tbody:
            for row in tbody.find_all("tr"):
                cells = row.find_all("td")
                if cells:
                    yield "| " + " | ".join(cell.get_text(strip=True) for cell in cells) + " |\n"
        
        yield "\n\n"


def create_html_parser() -> HTMLParser:
    """Create an HTML parser instance.
    
    Returns:
        HTMLParser instance
    """
    return HTMLParser()