from enum import Enum


class ResponseSignalEnum(Enum):
    """Machine-readable response/validation signals, mini-rag style."""

    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_VALIDATED_SUCCESS = "file_validated_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    FILE_UPLOAD_SUCCESS = "file_upload_success"

    DOCUMENT_NOT_FOUND = "document_not_found"
    NO_PENDING_CHUNKS = "no_pending_chunks"
    EMPTY_CHUNK_SET = "empty_chunk_set"
    EMBEDDING_FAILED = "embedding_failed"
    EMBEDDING_SUCCESS = "embedding_success"

    VECTORDB_SEARCH_ERROR = "vector_db_search_error"
    VECTORDB_SEARCH_SUCCESS = "vector_db_search_success"
    RAG_ANSWER_ERROR = "rag_answer_error"
    RAG_ANSWER_SUCCESS = "rag_answer_success"
