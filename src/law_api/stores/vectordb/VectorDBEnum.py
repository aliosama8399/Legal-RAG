from enum import Enum


class VectorDBEnum(Enum):
    QDRANT_LOCAL = "qdrant-local"
    QDRANT = "qdrant"
    PGVECTOR = "postgresql"
    IN_MEMORY_TEST = "in-memory-test"
