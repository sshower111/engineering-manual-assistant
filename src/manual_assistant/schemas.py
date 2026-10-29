"""Request and response models of the HTTP API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Citation(BaseModel):
    source: str = Field(description="URL of the cited page or PDF page")
    section: str = Field(description="Document title and section heading")
    snippet: str = Field(description="Start of the cited chunk")


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(
        ge=0.0, le=1.0, description="Retrieval similarity of the best cited chunk; 0 on refusal"
    )
