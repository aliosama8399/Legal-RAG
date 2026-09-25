"""SQLAlchemy ORM models mirroring the PostgreSQL/pgvector storage tables.

These mirror the schema created by the PostgreSQL provider
(``law_documents`` / ``law_chunks`` / ``law_chat_history``). The pgvector
``embedding`` column is intentionally omitted here — it is managed by raw
SQL in the provider, since its dimension is fixed at first embed.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LawDocument(Base):
    __tablename__ = "law_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(Text)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LawChunkRecord(Base):
    __tablename__ = "law_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("law_documents.id"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    article_number: Mapped[int] = mapped_column(Integer, nullable=False)
    citation: Mapped[str] = mapped_column(Text, nullable=False)
    source_page: Mapped[int] = mapped_column(Integer, nullable=False)
    text_ar: Mapped[str] = mapped_column(Text, nullable=False)
    text_en: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class LawChatHistory(Base):
    __tablename__ = "law_chat_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int | None] = mapped_column(Integer)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
