from pydantic import BaseModel, Field, validator
from typing import List, Dict, Literal, Optional
from datetime import datetime

class BlogSection(BaseModel):
    """Model for a blog post section"""
    type: Literal["introduction", "body", "conclusion"]
    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content")