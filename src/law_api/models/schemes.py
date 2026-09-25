"""Pydantic validation models for data flowing through the pipeline.

Used to validate ingested chunks before embedding and search results
returned from the vector store.
"""

from pydantic import BaseModel, ConfigDict, Field


class LawArticle(BaseModel):
    """A single bilingual Egyptian Civil Code article."""

    model_config = ConfigDict(extra="allow")

    article_number: int = Field(ge=1)
    text_en: str = ""
    text_ar: str = ""
    is_repealed: bool = False
    source_page: int = Field(default=0, ge=0)
    citation: str = Field(min_length=1)


class LawChunk(BaseModel):
    """A citation-preserving paragraph chunk of one article.

    Chunk payloads carry extra hierarchy fields (book/chapter/section/topic),
    so unknown fields are allowed through validation.
    """

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(min_length=1)
    article_number: int = Field(ge=1)
    chunk_text: str
    chunk_number: int = Field(ge=1)
    citation: str = Field(min_length=1)
    source_page: int = Field(ge=0)


class RetrievedDocument(BaseModel):
    """A vector-store search hit with similarity score."""

    model_config = ConfigDict(extra="allow")

    document_id: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    article_number: int = Field(ge=1)
    chunk_text: str
    citation: str
    source_page: int = Field(ge=0)
    score: float
