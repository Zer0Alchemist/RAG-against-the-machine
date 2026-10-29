from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """
    this model represents a source chunk with its original
    file path, starting and ending character indices and content
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str | None = Field(default=None)


class UnansweredQuestion(BaseModel):
    """
    this model represents an unanswered question containing
    a unique question identifier and the question text
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    this model extends UnansweredQuestion by including
    reference sources and the expected answer
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    this model represents a dataset containing a list
    of answered or unanswered RAG questions
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    this model represents the retrieval results for a question,
    including its identifier, text and retrieved source chunks
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    this model extends MinimalSearchResults by adding
    the generated answer for the question
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    this model represents a collection of student search results
    and the number of retrieved sources requested per question
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    this model represents a collection of generated answers
    with their retrieved sources and the retrieval k value
    """
    search_results: List[MinimalAnswer]
    k: int
