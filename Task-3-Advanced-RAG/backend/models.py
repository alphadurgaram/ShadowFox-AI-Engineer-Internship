from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the uploaded document"
    )


class Source(BaseModel):
    page: int
    text: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool