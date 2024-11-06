import json
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Optional, Type
from datetime import datetime

# Define models with field descriptions and JSON schema customization
class BaseContentOutput(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(),
                           description="The timestamp of when the content was generated or published.")
    content_type: str = Field(..., description="The type of content being generated (e.g., 'image_post', 'linkedin_post').")
    platform: str = Field(..., description="The platform where the content will be published (e.g., 'Instagram').")
    target_audience: Optional[str] = Field(None, description="Optional. The target audience for the content (e.g., 'German-speaking professionals').")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-11-06T14:50:13.910242",
                "content_type": "image_post",
                "platform": "Instagram",
                "target_audience": "German-speaking professionals"
            }
        }

class InstagramPostOutput(BaseContentOutput):
    caption: str = Field(..., max_length=125, description="The caption for the Instagram post, with a max length of 125 characters.")
    hashtags: List[str] = Field(..., min_items=1, max_items=30, description="A list of hashtags to include in the Instagram post (1-30 items).")
    image_prompt: str = Field(..., description="A prompt describing the image to be created for the Instagram post.")
    visual_elements: List[str] = Field(..., description="A list of visual elements that should appear in the post (e.g., 'product photo', 'brand logo').")
    mood: str = Field(..., description="The mood or tone of the post (e.g., 'exciting').")
    engagement_hooks: List[str] = Field(..., description="A list of hooks to encourage engagement (e.g., 'question', 'call to action').")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-11-06T14:50:13.910242",
                "content_type": "image_post",
                "platform": "Instagram",
                "target_audience": "German-speaking professionals",
                "caption": "Exciting new product launch! 🚀",
                "hashtags": ["#productlaunch", "#innovation"],
                "image_prompt": "Modern product photography with soft lighting",
                "visual_elements": ["product photo", "brand logo"],
                "mood": "exciting",
                "engagement_hooks": ["question", "call to action"]
            }
        }

class LinkedInPostOutput(BaseContentOutput):
    content: str = Field(..., description="The main content of the LinkedIn post.")
    title: str = Field(..., description="The title of the LinkedIn post.")
    key_insights: List[str] = Field(..., description="Key insights that should be highlighted in the post.")
    image_prompt: str = Field(..., description="A prompt describing the image to accompany the LinkedIn post.")
    call_to_action: str = Field(..., description="A call to action to be included at the end of the post.")
    hashtags: List[str] = Field(..., description="A list of hashtags to include in the LinkedIn post.")

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
                "image_prompt": "Futuristic AI tech imagery",
                "call_to_action": "Learn more about AI solutions",
                "hashtags": ["#AI", "#Innovation"]
            }
        }

class CompanyBlogOutput(BaseContentOutput):
    title: str = Field(..., description="The title of the blog post.")
    meta_description: str = Field(..., description="A short description of the blog post for SEO purposes.")
    keywords: List[str] = Field(..., description="SEO keywords related to the blog post.")
    word_count: int = Field(..., description="The word count of the blog post.")
    image_prompt: str = Field(..., description="A prompt describing the image to accompany the blog post.")
    seo_elements: Dict[str, str] = Field(..., description="A dictionary of SEO-related elements, such as meta tags and keywords.")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-11-06T14:50:13.910242",
                "content_type": "blog_post",
                "platform": "Company Blog",
                "title": "Understanding AI for Business",
                "meta_description": "This blog explains the importance of AI for modern businesses.",
                "keywords": ["AI", "business", "technology"],
                "word_count": 1200,
                "image_prompt": "AI and business integration visuals",
                "seo_elements": {"meta_title": "AI for Business", "meta_keywords": "AI, business"}
            }
        }

# Helper function for model selection
def parse_content_output(data: dict) -> BaseContentOutput:
    content_type = data.get("content_type")
    model_map: Dict[str, Type[BaseContentOutput]] = {
        "image_post": InstagramPostOutput,
        "linkedin_post": LinkedInPostOutput,
        "blog_post": CompanyBlogOutput,
    }

    model_cls = model_map.get(content_type)
    if model_cls is None:
        raise ValueError(f"Unknown content type: {content_type}")

    return model_cls.parse_obj(data)

# Example result dictionary
result_dict = {
    "timestamp": "2024-11-06T14:50:13.910242",
    "content_type": "image_post",
    "platform": "Instagram",
    "target_audience": "German-speaking professionals",
    "caption": "Exciting new product launch! 🚀",
    "hashtags": ["#productlaunch", "#innovation"],
    "image_prompt": "Modern product photography with soft lighting",
    "visual_elements": ["product photo", "brand logo"],
    "mood": "exciting",
    "engagement_hooks": ["question", "call to action"]
}

# Parse the result using the helper function
try:
    parsed_output = parse_content_output(result_dict)
    # Using json.dumps for pretty-printing
    print(json.dumps(parsed_output.dict(), indent=2))
except ValidationError as e:
    print(f"Validation error: {e}")
except ValueError as ve:
    print(f"Value error: {ve}")
