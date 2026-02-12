"""URL / web page loader for external knowledge base."""

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class URLLoader:
    """Loader for web pages. Fetches and extracts main text content."""

    def load(self, url: str) -> str:
        """Fetch a URL and extract readable text content."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, nav elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Prefer main content areas
        main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|article|post|main", re.I))
        root = main if main else soup.body or soup

        if not root:
            return ""

        text = root.get_text(separator="\n", strip=True)
        # Collapse multiple newlines
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def get_display_name(url: str) -> str:
        """Derive a short display name from URL (e.g. domain or path)."""
        parsed = urlparse(url)
        domain = parsed.netloc or "page"
        path = parsed.path.strip("/")
        if path and path != "/":
            # Use last path segment or domain
            name = path.split("/")[-1][:30] or domain
        else:
            name = domain.replace("www.", "")
        return name[:50]
