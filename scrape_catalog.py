import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE = "https://www.shl.com"
CATALOG_PATH = "/products/product-catalog/"
TYPE_INDIVIDUAL = 1  # Individual Test Solutions
PAGE_SIZE = 12

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class CatalogItem:
    id: Optional[str]
    name: str
    url: str
    description: str
    duration: int
    adaptive_support: str
    remote_support: str
    test_type: List[str]


def _ensure_data_dir() -> str:
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _bool_from_cell(cell) -> bool:
    if cell is None:
        return False

    # look for span with class "catalogue__circle" AND class "-yes"
    for span in cell.find_all("span"):
        classes = span.get("class", [])
        if "catalogue__circle" in classes and "-yes" in classes:
            return True

    return False


def _extract_test_type_letters(row) -> List[str]:
    # Typically: <span class="product-catalogue__key" ...>A</span>
    letters = []
    for sp in row.select("td.product-catalogue__keys span.product-catalogue__key"):
        t = sp.get_text(strip=True)
        if t:
            letters.append(t)
    # sometimes keys are not under that exact class, fallback:
    if not letters:
        for sp in row.select("span.product-catalogue__key"):
            t = sp.get_text(strip=True)
            if t:
                letters.append(t)
    return letters


def _parse_list_page(soup: BeautifulSoup) -> List[Tuple[str, str, bool, bool, List[str], Optional[str]]]:
    """
    Returns list of tuples:
    (name, absolute_url, remote_yes, adaptive_yes, test_type_letters, entity_id)
    """
    results = []

    # Rows have data-entity-id in view-source screenshot
    rows = soup.select("tr[data-entity-id]")
    for row in rows:
        entity_id = row.get("data-entity-id")
        a = row.select_one("td.custom__table-heading__title a[href]")
        if not a:
            continue
        name = a.get_text(strip=True)
        href = a.get("href", "").strip()
        if not href:
            continue
        abs_url = urljoin(BASE, href)

        tds = row.find_all("td")
        # Expected structure (from header):
        # 0 = title, 1 = Remote Testing, 2 = Adaptive/IRT, 3 = Test Type
        remote_yes = _bool_from_cell(tds[1] if len(tds) > 1 else None)
        adaptive_yes = _bool_from_cell(tds[2] if len(tds) > 2 else None)
        test_types = _extract_test_type_letters(row)

        results.append((name, abs_url, remote_yes, adaptive_yes, test_types, entity_id))

    return results


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def _extract_duration_minutes(text: str) -> int:
    """
    Extract duration in minutes from various patterns:
    - 30 minutes
    - 30 mins
    - minutes = 30
    - Completion Time in minutes = 30
    """
    if not text:
        return 0

    text = text.lower()

    # Pattern 1: "30 minutes" / "30 mins"
    m = re.search(r"\b(\d{1,3})\s*(minutes|minute|mins|min)\b", text)
    if m:
        return int(m.group(1))

    # Pattern 2: "minutes = 30"
    m = re.search(r"minutes?\s*=\s*(\d{1,3})", text)
    if m:
        return int(m.group(1))

    return 0


def _parse_detail_page(soup: BeautifulSoup) -> Tuple[str, int]:
    """
    Extract description and duration (if present).
    SHL pages vary; we try multiple selectors.
    """
    # Try to find a main content container
    # Common pattern: article / main / div with rich text
    candidates = []

    main = soup.select_one("main")
    if main:
        candidates.append(main)

    article = soup.select_one("article")
    if article:
        candidates.append(article)

    # If neither, fallback to body
    candidates.append(soup)

    full_text = ""
    for c in candidates:
        # Prefer paragraphs and list items for description
        parts = [p.get_text(" ", strip=True) for p in c.select("p")[:20]]
        if parts:
            full_text = " ".join(parts)
            break

    if not full_text:
        # last resort: all text (trim)
        full_text = soup.get_text(" ", strip=True)

    full_text = _clean_text(full_text)

    # A reasonable description: first ~1200 chars
    description = full_text[:1200]

    duration = _extract_duration_minutes(full_text)

    return description, duration


def scrape_catalog(output_path: str = os.path.join("data", "catalog.json")) -> List[CatalogItem]:
    _ensure_data_dir()

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    seen_urls: Set[str] = set()
    items: List[CatalogItem] = []

    start = 0
    page_num = 1

    while True:
        list_url = f"{BASE}{CATALOG_PATH}?start={start}&type={TYPE_INDIVIDUAL}"
        print(f"[list] page={page_num} start={start} url={list_url}")

        soup = _get_soup(session, list_url)
        rows = _parse_list_page(soup)

        if not rows:
            print(f"[done] No rows found at start={start}. Stopping.")
            break

        for (name, abs_url, remote_yes, adaptive_yes, test_types, entity_id) in rows:
            if abs_url in seen_urls:
                continue
            seen_urls.add(abs_url)

            # polite delay
            time.sleep(0.15)

            try:
                detail_soup = _get_soup(session, abs_url)
                description, duration = _parse_detail_page(detail_soup)
            except Exception as e:
                print(f"[warn] detail fetch failed: {abs_url} err={e}")
                description, duration = "", 0

            item = CatalogItem(
                id=entity_id,
                name=name,
                url=abs_url,
                description=description,
                duration=duration,
                adaptive_support="Yes" if adaptive_yes else "No",
                remote_support="Yes" if remote_yes else "No",
                test_type=test_types,
            )
            items.append(item)

        start += PAGE_SIZE
        page_num += 1
        time.sleep(0.25)

    # Save JSON
    out = [asdict(x) for x in items]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[saved] {output_path}")
    print(f"[count] total_items={len(items)}")
    return items


if __name__ == "__main__":
    scrape_catalog()
