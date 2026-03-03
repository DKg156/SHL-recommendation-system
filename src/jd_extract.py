import re
from typing import Optional

import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url: str, timeout: int = 20, max_chars: int = 8000) -> str:
    """
    Fetch a public JD page and extract readable text.
    Raises ValueError on bad/empty extraction.
    """
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError("Invalid URL. Please provide a public http/https URL.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove noisy elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    # Prefer main/article content if present
    root = soup.find("main") or soup.find("article") or soup.body or soup

    text = root.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    if not text or len(text) < 80:
        raise ValueError("Could not extract enough text from the provided URL.")

    return text[:max_chars]
