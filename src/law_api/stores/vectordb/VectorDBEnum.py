from enum import Enum


class VectorDBEnum(Enum):
    QDRANT_LOCAL = "qdrant-local"
    QDRANT = "qdrant"
    PGVECTOR = "postgresql"
