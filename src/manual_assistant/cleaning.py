"""Turn raw downloads (wiki HTML, manual PDF) into cleaned ``Section`` objects."""

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from manual_assistant.chunking import normalize_whitespace
from manual_assistant.models import Section

# Wiki elements that are navigation or chrome rather than content.
WIKI_JUNK_SELECTORS = [
    "script",
    "style",
    ".mw-editsection",
    ".mw-pt-languages",
    ".docnav",
    ".toc",
    "#toc",
    ".noprint",
    ".navbox",
    "sup.reference",
    ".reference",
    ".references",
]
HEADING_TAGS = {"h2", "h3", "h4"}
INTRO_SECTION = "Introduction"


def _heading_of(element: Tag) -> Tag | None:
    """Return the heading tag if ``element`` is one (old or new MediaWiki markup)."""
    if element.name in HEADING_TAGS:
        return element
    if element.name == "div" and "mw-heading" in (element.get("class") or []):
        return element.find(list(HEADING_TAGS))
    return None


def _heading_anchor(heading: Tag) -> str:
    if heading.get("id"):
        return str(heading["id"])
    headline = heading.find(class_="mw-headline")
    if headline is not None and headline.get("id"):
        return str(headline["id"])
    return ""


def wiki_html_to_sections(html: str, title: str, page_url: str) -> list[Section]:
    """Split a rendered wiki page into one ``Section`` per heading."""
    soup = BeautifulSoup(html, "html.parser")
    for selector in WIKI_JUNK_SELECTORS:
        for element in soup.select(selector):
            element.decompose()
    root = soup.select_one(".mw-parser-output") or soup

    sections: list[Section] = []
    name, url, parts = INTRO_SECTION, page_url, []

    def flush() -> None:
        text = normalize_whitespace("\n\n".join(parts))
        if text:
            sections.append(Section("freecad", title, name, url, text))

    for element in root.children:
        if not isinstance(element, Tag):
            if str(element).strip():
                parts.append(str(element))
            continue
        heading = _heading_of(element)
        if heading is not None:
            flush()
            name = heading.get_text(" ", strip=True)
            anchor = _heading_anchor(heading)
            url = f"{page_url}#{anchor}" if anchor else page_url
            parts = []
        else:
            parts.append(element.get_text(" ", strip=True))
    flush()
    return sections


@dataclass(frozen=True)
class OutlineEntry:
    path: tuple[str, ...]  # titles from the top level down to this entry
    page: int  # zero-based page index where the entry starts


def _find_title(page_text: str, title: str, start: int) -> int:
    """Position of ``title`` in ``page_text`` at or after ``start``, or -1."""
    return page_text.lower().find(title.lower(), start)


def pdf_pages_to_sections(
    pages: list[str], outline: list[OutlineEntry], doc_title: str, pdf_url: str
) -> list[Section]:
    """Cut the page texts of a PDF into sections using its outline (bookmarks).

    Each outline entry starts where its title first appears on its start page
    (or at the top of the page if the title text cannot be found) and runs until
    the next entry starts. Text before the first entry is ignored (cover, TOC).
    """
    entries = sorted(outline, key=lambda e: e.page)
    # Resolve every entry to an absolute (page, offset) position.
    positions: list[tuple[int, int]] = []
    for entry in entries:
        page_text = pages[entry.page] if entry.page < len(pages) else ""
        same_page_before = [off for page, off in positions if page == entry.page]
        search_from = max(same_page_before, default=0)
        offset = _find_title(page_text, entry.path[-1], search_from)
        positions.append((entry.page, offset if offset >= 0 else search_from))

    sections = []
    for i, entry in enumerate(entries):
        start_page, start_off = positions[i]
        end_page, end_off = positions[i + 1] if i + 1 < len(entries) else (len(pages) - 1, None)
        texts = []
        for page in range(start_page, end_page + 1):
            text = pages[page]
            begin = start_off if page == start_page else 0
            stop = end_off if page == end_page else None
            texts.append(text[begin:stop])
        text = normalize_whitespace("\n".join(texts))
        if text:
            label = " > ".join(entry.path[-2:])
            url = f"{pdf_url}#page={start_page + 1}"
            sections.append(Section("calculix", doc_title, label, url, text))
    return sections


def read_pdf(path: str) -> tuple[list[str], list[OutlineEntry]]:
    """Extract page texts and a flattened outline from a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    outline: list[OutlineEntry] = []

    def walk(items: list, parents: tuple[str, ...]) -> None:
        last: tuple[str, ...] = parents
        for item in items:
            if isinstance(item, list):
                walk(item, last)
                continue
            path = (*parents, str(item.title).strip())
            page = reader.get_destination_page_number(item)
            if page is not None and page >= 0:
                outline.append(OutlineEntry(path, page))
            last = path

    walk(reader.outline, ())
    return pages, outline
