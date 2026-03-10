from pydantic import BaseModel, field_validator
from typing import Optional, List


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value

    @field_validator("k")
    @classmethod
    def k_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError("k must be greater than 0")
        return value

class SourceDocument(BaseModel):
    source: str
    page: int
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    processing_time_ms: float

class IngestRequest(BaseModel):
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class IngestResponse(BaseModel):
    pages_loaded: int
    chunks_created: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool