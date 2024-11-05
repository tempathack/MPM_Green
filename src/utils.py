from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from datetime import datetime

class SummaryOutput(BaseModel):
    """Pydantic model for structured summary output"""
    title: str = Field(
        description="A fitting and engaging title that reflects the essence of the text"
    )
    summary: str = Field(
        description="A comprehensive summary of the text between 100-150 words"
    )
    key_points: List[str] = Field(
        description="Main points extracted from the text",
        min_items=3,
        max_items=7
    )
    word_count: int = Field(
        description="The word count of the summary"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @validator('summary')
    def validate_summary_length(cls, v):
        words = len(v.split())
        if not 100 <= words <= 150:
            raise ValueError(f'Summary must be between 100-150 words, got {words} words')
        return v

    @validator('title')
    def validate_title_length(cls, v):
        words = len(v.split())
        if words > 15:
            raise ValueError(f'Title must not exceed 15 words, got {words} words')
        return v


def setup_summarization_chain():
    """Setup the summarization chain with Pydantic parser"""
    # Initialize parser
    parser = PydanticOutputParser(pydantic_object=SummaryOutput)

    # Initialize model
    model = ChatOpenAI(temperature=0, model="gpt-4o")

    # Create prompt template with format instructions
    prompt = ChatPromptTemplate.from_template("""
    ### Instructions:
    As an expert in text summarization, analyze the provided text and generate a structured response.

    {format_instructions}

    ### Text to Summarize:
    {text}

    ### Requirements:
    1. Create a compelling title (max 15 words)
    2. Write a comprehensive summary (100-150 words)
    3. Extract 3-7 key points
    4. Maintain objective tone
    5. Ensure clarity and coherence

    ### Response Format:
    Provide the output in a structured format that matches the specified schema.
    """)

    # Combine components
    chain = (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | model
            | parser
    )

    return chain