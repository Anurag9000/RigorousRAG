import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from typing import Optional

# Configuration - could be moved to a central config later
DEFAULT_USER_AGENT = "AcademicSearchBot/2.0 (+https://example.com/academic-search-bot-info)"
REQUEST_TIMEOUT = 10
MAX_CONTENT_LENGTH = 5_000_000 # 5MB

class PageContent(BaseModel):
    url: str
    title: str
    text: str
    content_length: int
    error: Optional[str] = None

def fetch_single_page(url: str, user_agent: str = DEFAULT_USER_AGENT) -> PageContent:
    """
    Fetches a single web page and extracts its content (title and text).
    """
    try:
        headers = {"User-Agent": user_agent}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Basic size check
        content_length_header = response.headers.get("Content-Length")
        if content_length_header and int(content_length_header) > MAX_CONTENT_LENGTH:
             return PageContent(
                url=url,
                title="Error",
                text="",
                content_length=int(content_length_header),
                error="Page too large"
            )

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Cleanup
        for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            element.decompose()

        # Extract title
        title = "Untitled"
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split()) # Normalize whitespace

        return PageContent(
            url=url,
            title=title,
            text=text,
            content_length=len(text)
        )

    except Exception as e:
        return PageContent(
            url=url,
            title="Error",
            text="",
            content_length=0,
            error=str(e)
        )
