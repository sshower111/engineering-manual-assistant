"""Plain data types shared by ingestion and retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """A cleaned section of a source document."""

    source: str  # "freecad" or "calculix"
    title: str  # page or document title
    section: str  # section heading inside the page
    url: str  # link to the section (anchor or #page=N)
    text: str


@dataclass(frozen=True)
class Chunk:
    """A piece of a section that gets embedded and stored."""

    id: str
    text: str
    source: str
    title: str
    section: str
    url: str

    def metadata(self) -> dict[str, str]:
        return {
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "url": self.url,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float  # cosine similarity, higher is better
    title: str
    section: str
    url: str
