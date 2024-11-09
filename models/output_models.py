import re
from pydantic import BaseModel, Field, validator, model_validator,field_validator
from typing import List, Dict, Literal, Optional, Union
from datetime import datetime


class BlogSection(BaseModel):
    """Model for a blog post section"""
    type: Literal["introduction", "body", "conclusion"]
    heading: str = Field(description="Section heading", min_length=1, max_length=100)
    content: str = Field(description="Section content", min_length=50)

    @field_validator('heading')
    def validate_heading_format(cls, v):
        if not v.strip():
            raise ValueError("Heading cannot be empty or just whitespace")
        return v.strip()

    @field_validator('content')
    def validate_content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty or just whitespace")
        return v.strip()


class BaseContentOutput(BaseModel):
    """Base class for all content outputs"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    content_type: str = Field(description="Type of content being generated")
    platform: str = Field(description="Platform this content is intended for")
    target_audience: Optional[str] = Field(description="Target audience for this content")
    pydantic_object: Optional[str] = None

    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("Invalid ISO format timestamp")
        return v

    @model_validator(mode="before")
    def set_defaults(cls, values):
        if values.get("pydantic_object") is None:
            values["pydantic_object"] = f"{values.get('platform', 'unknown')}-{datetime.now().isoformat()}"
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-11-06T14:50:13.910242",
                "content_type": "image_post",
                "platform": "Instagram",
                "target_audience": "German-speaking professionals"
            }
        }


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
    visual_elements: List[str] = Field(min_items=1)
    mood: str = Field(description="Overall mood/tone")
    engagement_hooks: List[str] = Field(min_items=1)

    @field_validator('caption')
    def validate_caption_format(cls, v):
        if len(v) > 125:
            raise ValueError(f'Caption must not exceed 125 characters, got {len(v)}')
        if not v.strip():
            raise ValueError("Caption cannot be empty or just whitespace")
        return v.strip()

    @field_validator('hashtags')
    def validate_hashtags(cls, v):
        cleaned_tags = []
        for tag in v:
            # Remove any spaces and special characters
            clean_tag = re.sub(r'[^\w]', '', tag.strip())
            if not clean_tag:
                continue
            cleaned_tags.append(f"#{clean_tag}")
        if not cleaned_tags:
            raise ValueError("Must provide at least one valid hashtag")
        return cleaned_tags

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-11-06T14:50:13.910242",
                "content_type": "image_post",
                "platform": "Instagram",
                "target_audience": "German-speaking professionals",
                "caption": "Exciting new product launch! 🚀",
                "hashtags": ["#productlaunch", "#innovation"],
                "visual_elements": ["product photo", "brand logo"],
                "mood": "exciting",
                "engagement_hooks": ["question", "call to action"]
            }
        }


class LinkedInPostOutput(BaseContentOutput):
    """Pydantic model for LinkedIn post output"""
    content: str = Field(max_length=3000)
    title: str = Field(max_length=100)
    key_insights: List[str] = Field(min_items=1, max_items=5)
    call_to_action: str = Field(min_length=5, max_length=150)
    hashtags: List[str] = Field(min_items=1, max_items=10)
    interactive_elements: Optional[Dict[str, Union[str, List[str]]]] = Field(
        default=None,
        description="Interactive elements like polls or questions"
    )

    @field_validator('content')
    def validate_content_format(cls, v):
        if len(v) > 3000:
            raise ValueError('Content must not exceed 3000 characters')
        if not v.strip():
            raise ValueError("Content cannot be empty or just whitespace")
        return v.strip()

    @field_validator('hashtags')
    def validate_hashtags(cls, v):
        cleaned_tags = []
        for tag in v:
            clean_tag = re.sub(r'[^\w]', '', tag.strip())
            if not clean_tag:
                continue
            cleaned_tags.append(f"#{clean_tag}")
        if not cleaned_tags:
            raise ValueError("Must provide at least one valid hashtag")
        return cleaned_tags

    @field_validator('interactive_elements')
    def validate_interactive_elements(cls, v):
        if v is None:
            return v
        allowed_types = {'poll', 'question', 'survey'}
        for key in v.keys():
            if key not in allowed_types:
                raise ValueError(f"Interactive element type must be one of {allowed_types}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-11-06T14:50:13.910242",
                "content_type": "linkedin_post",
                "platform": "LinkedIn",
                "target_audience": "Business professionals",
                "content": "Check out our latest innovation in AI.",
                "title": "AI Innovation for 2024",
                "key_insights": ["AI for business", "Revolutionizing tech"],
                "call_to_action": "Learn more about AI solutions",
                "hashtags": ["#AI", "#Innovation"],
                "interactive_elements": {
                    "poll": ["What do you think about AI?"],
                    "question": ["How is AI transforming your business?"]
                }
            }
        }


class CompanyBlogOutput(BaseContentOutput):
    """Pydantic model for blog post output"""
    title: str = Field(max_length=100)
    meta_description: str = Field(max_length=160)
    keywords: List[str] = Field(min_items=3, max_items=10)
    sections: List[BlogSection]
    word_count: int = Field(ge=600, le=2000)
    seo_elements: Dict[str, str] = Field(
        description="SEO elements including meta tags, header tags, etc."
    )
    read_time: Optional[int] = None

    @field_validator('word_count')
    def validate_word_count(cls, v):
        if not 600 <= v <= 2000:
            raise ValueError('Blog post must be between 600-2000 words')
        return v

    @field_validator('sections')
    def validate_sections(cls, v):
        section_types = {section.type for section in v}
        required_types = {'introduction', 'body', 'conclusion'}
        if not required_types.issubset(section_types):
            raise ValueError(f'Missing required sections: {required_types}')
        return v

    @model_validator(mode="after")
    def calculate_read_time(cls, model):
        if model.word_count:
            # Assuming average reading speed of 200 words per minute
            model.read_time = round(model.word_count / 200)
        return model

    @field_validator('seo_elements')
    def validate_seo_elements(cls, v):
        required_elements = {'title_tag', 'meta_description', 'canonical_url'}
        missing_elements = required_elements - set(v.keys())
        if missing_elements:
            raise ValueError(f"Missing required SEO elements: {missing_elements}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-11-06T14:50:13.910242",
                "content_type": "blog_post",
                "platform": "Company Blog",
                "target_audience": "Content marketers",
                "title": "The Importance of SEO",
                "meta_description": "Learn the latest SEO strategies to improve your website's ranking.",
                "keywords": ["SEO", "digital marketing", "website optimization"],
                "sections": [
                    {"type": "introduction", "heading": "Why SEO Matters", "content": "SEO is essential for businesses..."},
                    {"type": "body", "heading": "How to Improve Your SEO", "content": "Here are some key strategies..."},
                    {"type": "conclusion", "heading": "Final Thoughts", "content": "SEO is an ongoing effort..."}
                ],
                "word_count": 1200,
                "seo_elements": {"title_tag": "SEO Strategies", "meta_description": "Learn SEO", "canonical_url": "https://example.com/blog/seo-strategies"},
                "read_time": 6
            }
        }
