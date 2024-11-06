from pydantic import BaseModel, Field, validator
from typing import List, Dict, Literal, Optional
from datetime import datetime

class BlogSection(BaseModel):
    """Model for a blog post section"""
    type: Literal["introduction", "body", "conclusion"]
    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content")

class BaseContentOutput(BaseModel):
    """Base class for all content outputs"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    generated_image: Optional[Dict[str, str]] = Field(
        description="Generated image data including URL and prompts",
        default=None
    )

class InstagramPostOutput(BaseContentOutput):
    """Pydantic model for Instagram post output"""
    caption: str = Field(
        description="Main caption text",
        max_length=125
    )
    hashtags: List[str] = Field(
        description="List of relevant hashtags",
        min_items=1,
        max_items=30
    )
    image_prompt: str = Field(description="Image generation prompt")
    visual_elements: List[str] = Field(min_items=1)
    mood: str = Field(description="Overall mood/tone")
    engagement_hooks: List[str] = Field(min_items=1)

    @validator('caption')
    def validate_caption_length(cls, v):
        if len(v) > 125:
            raise ValueError(f'Caption must not exceed 125 characters, got {len(v)}')
        return v

    @validator('hashtags')
    def validate_hashtags(cls, v):
        return [f"#{tag.lstrip('#')}" for tag in v]

class LinkedInPostOutput(BaseContentOutput):
    """Pydantic model for LinkedIn post output"""
    content: str = Field(max_length=3000)
    title: str = Field(max_length=100)
    key_insights: List[str] = Field(min_items=1, max_items=5)
    image_prompt: str
    call_to_action: str
    hashtags: List[str] = Field(min_items=1, max_items=10)
    interactive_elements: Optional[dict] = None

    @validator('content')
    def validate_content_length(cls, v):
        if len(v) > 3000:
            raise ValueError('Content must not exceed 3000 characters')
        return v

    @validator('hashtags')
    def validate_hashtags(cls, v):
        return [f"#{tag.lstrip('#')}" for tag in v]

class CompanyBlogOutput(BaseContentOutput):
    """Pydantic model for blog post output"""
    title: str = Field(max_length=100)
    meta_description: str = Field(max_length=160)
    keywords: List[str] = Field(min_items=3, max_items=10)
    sections: List[BlogSection]
    word_count: int
    image_prompt: str
    seo_elements: Dict[str, str]

    @validator('word_count')
    def validate_word_count(cls, v):
        if not 600 <= v <= 2000:
            raise ValueError('Blog post must be between 600-2000 words')
        return v

    @validator('sections')
    def validate_sections(cls, v):
        section_types = {section.type for section in v}
        required_types = {'introduction', 'body', 'conclusion'}
        if not required_types.issubset(section_types):
            raise ValueError(f'Missing required sections: {required_types}')
        return v