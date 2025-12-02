"""Document loaders for various data sources."""

from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from loguru import logger
import xml.etree.ElementTree as ET
from pathlib import Path

from parsers.html_parser import create_html_parser
from config.settings import settings


class DocumentLoader:
    """Base class for document loaders."""
    
    def __init__(self):
        """Initialize the document loader."""
        self.parser = create_html_parser()
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from the source.
        
        Returns:
            List of document dictionaries
        """
        raise NotImplementedError


class SitemapLoader(DocumentLoader):
    """Loader for sitemap-based document sources."""
    
    def __init__(self, sitemap_url: str, base_url: str, document_type: str):
        """Initialize sitemap loader.
        
        Args:
            sitemap_url: URL of the sitemap
            base_url: Base URL for the site
            document_type: Type of documents being loaded
        """
        super().__init__()
        self.sitemap_url = sitemap_url
        self.base_url = base_url
        self.document_type = document_type
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from sitemap.
        
        Returns:
            List of document dictionaries
        """
        try:
            logger.info(f"Loading documents from sitemap: {self.sitemap_url}")
            
            # Parse sitemap
            urls = self._parse_sitemap()
            logger.info(f"Found {len(urls)} URLs in sitemap")
            
            # Load documents from URLs
            documents = []
            for url in urls:
                try:
                    doc = self._load_document_from_url(url)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load document from {url}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from sitemap: {e}")
            raise
    
    def _parse_sitemap(self) -> List[str]:
        """Parse sitemap XML and extract URLs.
        
        Returns:
            List of URLs
        """
        import requests
        
        response = requests.get(self.sitemap_url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        urls = []
        
        # Handle different sitemap formats
        for url_elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc_elem = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            if loc_elem is not None:
                url = loc_elem.text
                if url and url.startswith(self.base_url):
                    urls.append(url)
        
        return urls
    
    def _load_document_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Load a single document from URL.
        
        Args:
            url: URL to load
            
        Returns:
            Document dictionary or None if failed
        """
        import requests
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            content = self.parser.parse_html(response.text, url)
            
            # Extract metadata
            soup = BeautifulSoup(response.text, "lxml")
            title = self._extract_title(soup)
            description = self._extract_description(soup)
            language = self._extract_language(soup)
            
            return {
                "content": content,
                "metadata": {
                    "source": url,
                    "title": title,
                    "description": description,
                    "language": language,
                    "document_type": self.document_type
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to load document from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Title text
        """
        title_elem = soup.find("title")
        return title_elem.get_text().strip() if title_elem else ""
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract description from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Description text
        """
        desc_elem = soup.find("meta", attrs={"name": "description"})
        return desc_elem.get("content", "").strip() if desc_elem else ""
    
    def _extract_language(self, soup: BeautifulSoup) -> str:
        """Extract language from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Language code
        """
        html_elem = soup.find("html")
        return html_elem.get("lang", "").strip() if html_elem else ""


class RecursiveURLLoader(DocumentLoader):
    """Loader for recursive URL crawling."""
    
    def __init__(self, base_url: str, max_depth: int, document_type: str):
        """Initialize recursive URL loader.
        
        Args:
            base_url: Base URL to start crawling
            max_depth: Maximum crawl depth
            document_type: Type of documents being loaded
        """
        super().__init__()
        self.base_url = base_url
        self.max_depth = max_depth
        self.document_type = document_type
        self.visited_urls = set()
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents by recursive crawling.
        
        Returns:
            List of document dictionaries
        """
        try:
            logger.info(f"Starting recursive crawl from {self.base_url}")
            
            documents = []
            urls_to_visit = [(self.base_url, 0)]
            
            while urls_to_visit:
                url, depth = urls_to_visit.pop(0)
                
                if url in self.visited_urls or depth > self.max_depth:
                    continue
                
                self.visited_urls.add(url)
                
                try:
                    doc = self._load_document_from_url(url)
                    if doc:
                        documents.append(doc)
                        
                        # Find new URLs to visit
                        if depth < self.max_depth:
                            new_urls = self._extract_urls_from_document(doc["content"], url)
                            for new_url in new_urls:
                                if new_url not in self.visited_urls:
                                    urls_to_visit.append((new_url, depth + 1))
                    
                except Exception as e:
                    logger.warning(f"Failed to process {url}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in recursive crawling: {e}")
            raise
    
    def _load_document_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Load a single document from URL."""
        # Similar to SitemapLoader implementation
        import requests
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content = self.parser.parse_html(response.text, url)
            soup = BeautifulSoup(response.text, "lxml")
            
            return {
                "content": content,
                "metadata": {
                    "source": url,
                    "title": self._extract_title(soup),
                    "description": self._extract_description(soup),
                    "language": self._extract_language(soup),
                    "document_type": self.document_type
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to load document from {url}: {e}")
            return None
    
    def _extract_urls_from_document(self, content: str, base_url: str) -> List[str]:
        """Extract URLs from document content."""
        # Implementation for extracting URLs from content
        # This is a simplified version - you might want to use a more sophisticated approach
        import re
        
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        
        # Filter URLs to only include those from the same domain
        base_domain = urlparse(base_url).netloc
        filtered_urls = []
        
        for url in urls:
            if urlparse(url).netloc == base_domain:
                filtered_urls.append(url)
        
        return filtered_urls


def create_document_loaders() -> List[DocumentLoader]:
    """Create all document loaders.
    
    Returns:
        List of document loader instances
    """
    loaders = [
        SitemapLoader(
            sitemap_url=settings.langchain_sitemap_url,
            base_url="https://python.langchain.com/",
            document_type="langchain_docs"
        ),
        SitemapLoader(
            sitemap_url=settings.langgraph_sitemap_url,
            base_url="https://langchain-ai.github.io/langgraph/",
            document_type="langgraph_docs"
        ),
        RecursiveURLLoader(
            base_url=settings.langsmith_base_url,
            max_depth=3,
            document_type="langsmith_docs"
        ),
        RecursiveURLLoader(
            base_url=settings.api_docs_base_url,
            max_depth=3,
            document_type="api_docs"
        )
    ]
    
    return loaders