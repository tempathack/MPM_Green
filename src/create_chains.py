from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Literal
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import Literal, Union, Dict, Optional,List
import logging

import re
load_dotenv()

class BlogSection(BaseModel):
    """Model for a blog post section"""
    type: Literal["introduction", "body", "conclusion"]
    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content")
_=load_dotenv()

# Platform Descriptions
INSTAGRAM_DESCRIPTION ="""### Instructions for Instagram Post Creation

As an expert Instagram post creator, your task is to craft an engaging Instagram post that captivates audiences through visual appeal and concise, compelling captions. Your caption should be no more than 125 characters to maximize engagement. Additionally, you will generate a detailed prompt for an image generator API to create a visually striking image that complements your post.

### Context and Requirements

- **Post Focus**: Visually appealing and succinct content.
- **Caption Length**: Up to 125 characters.
- **Image Description**: Provide a detailed prompt for an image generator to ensure the image aligns perfectly with the post content.

### Desired Outcome

Create a captivating Instagram post with a short, engaging caption and a matching image prompt. The image should be vibrant, eye-catching, and directly related to the theme of your post.

---

**Example Post:**

Caption: "Sunset vibes 🌅✨ Embrace the evening glow and let your spirit shine. #SunsetMagic"

Image Generator API Prompt: "Create a breathtaking scene of a sunset over a tranquil beach, with vibrant orange and pink hues reflecting on calm waters. Include silhouettes of palm trees gently swaying in the breeze, and a clear sky with a few scattered clouds to enhance the sunset's beauty."
"""

COMPANY_BLOG_DESCRIPTION  = """### Expert Blog Post and Visual Creation for a Company Blog

As an expert company blog post writer, your task is to create a comprehensive and SEO-optimized article that delivers valuable insights and maintains proper structure.

### Required Structure:
1. Each blog post must contain:
   - Introduction section (type: "introduction")
   - At least one body section (type: "body")
   - Conclusion section (type: "conclusion")

2. Each section must have:
   - A clear heading
   - Detailed content
   - Proper flow and transitions

### Requirements:
- Total word count: 600-2000 words
- SEO-optimized content with relevant keywords
- Professional tone with engaging style
- Clear section organization

### Example Format:
{
    "title": "Your SEO-Optimized Title",
    "meta_description": "Compelling meta description under 160 characters",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "sections": [
        {
            "type": "introduction",
            "heading": "Introduction",
            "content": "Opening content..."
        },
        {
            "type": "body",
            "heading": "Main Section Title",
            "content": "Main content..."
        },
        {
            "type": "conclusion",
            "heading": "Conclusion",
            "content": "Concluding thoughts..."
        }
    ],
    "word_count": 800,
    "image_prompt": "Detailed image generation prompt",
    "seo_elements": {
        "title_tag": "SEO Title Tag",
        "meta_description": "SEO Meta Description"
    }
}

### Special Instructions:
1. Maintain proper section types (introduction, body, conclusion)
2. Ensure sufficient word count (minimum 600 words)
3. Create engaging section headings
4. Include relevant SEO elements
5. Generate appropriate image prompt

Please structure your response exactly according to this format, ensuring all required fields and sections are included."""

LINKEDIN_DESCRIPTION = """### Instructions for LinkedIn Expert Post Creation

As a LinkedIn expert post writer, your task is to craft an engaging and professional LinkedIn post that adheres to the following guidelines:

1. **Content and Style**: Your post should blend professional insights with a visually appealing format. Focus on sharing authentic stories or showcasing industry expertise. Ensure the tone is professional yet approachable.

2. **Length and Structure**: Aim for a post length of up to 3,000 characters. Structure the post to include a compelling introduction, informative content, and a thought-provoking conclusion or call-to-action.

3. **Visual Elements**: Enhance your post's engagement by incorporating a relevant image. Use dimensions of 1200 x 627 pixels for optimal display.

4. **Engagement Strategies**: Consider integrating video content or interactive elements like polls to boost engagement rates.

5. **Image Generation**: After crafting the post, create a prompt for an image generator API that will produce an image complementing your post content. The image should visually represent the core message or theme of your post.

### Example LinkedIn Post

"Navigating today's fast-paced digital landscape requires not just skills but a mindset geared towards continuous learning. 🌟

In my journey as a [Your Profession/Industry], I've learned that embracing change and fostering innovation are key to staying ahead. Whether it's adapting to new technologies or refining our strategies, growth begins with an open mind.

Join me as I explore the latest trends in [Industry], and let's discuss how we can harness these insights to drive success. Your thoughts?"

### Image Generator API Prompt

"Generate an image depicting a dynamic and futuristic digital landscape, symbolizing innovation and growth in the [Industry] sector. Use modern design elements and a vibrant color palette to convey progress and forward-thinking."""

PLATFORM_PROMPTS={}
PLATFORM_PROMPTS['linkedin']=LINKEDIN_DESCRIPTION
PLATFORM_PROMPTS['blog']=COMPANY_BLOG_DESCRIPTION
PLATFORM_PROMPTS['instagram']=INSTAGRAM_DESCRIPTION

class InstagramPostOutput(BaseModel):
    """Pydantic model for Instagram post output"""
    caption: str = Field(
        description="Main caption text for the Instagram post",
        max_length=125
    )
    hashtags: List[str] = Field(
        description="List of relevant hashtags",
        min_items=1,
        max_items=30
    )
    image_prompt: str = Field(
        description="Detailed prompt for image generation API"
    )
    visual_elements: List[str] = Field(
        description="List of key visual elements to include",
        min_items=1
    )
    mood: str = Field(
        description="Overall mood/tone of the post"
    )
    engagement_hooks: List[str] = Field(
        description="Engagement-driving elements in the post",
        min_items=1
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @validator('caption')
    def validate_caption_length(cls, v):
        if len(v) > 125:
            raise ValueError(f'Caption must not exceed 125 characters, got {len(v)}')
        return v

    @validator('hashtags')
    def validate_hashtags(cls, v):
        for hashtag in v:
            if not hashtag.startswith('#'):
                raise ValueError(f'Hashtag must start with #: {hashtag}')
        return v
class CompanyBlogOutput(BaseModel):
    """Updated Pydantic model for company blog post output"""
    title: str = Field(
        description="Blog post title",
        max_length=100
    )
    meta_description: str = Field(
        description="SEO meta description",
        max_length=160
    )
    keywords: List[str] = Field(
        description="SEO keywords",
        min_items=3,
        max_items=10
    )
    sections: List[BlogSection] = Field(
        description="Blog post sections with type, heading, and content",
        min_items=3
    )
    word_count: int = Field(
        description="Total word count of the blog post"
    )
    image_prompt: str = Field(
        description="Detailed prompt for feature image generation"
    )
    seo_elements: Dict[str, str] = Field(
        description="SEO-related elements including title tag, meta description"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @validator('word_count')
    def validate_word_count(cls, v):
        if not 600 <= v <= 2000:
            raise ValueError(f'Blog post must be between 600-2000 words, got {v}')
        return v

    @validator('sections')
    def validate_sections(cls, v):
        section_types = {section.type for section in v}
        required_types = {'introduction', 'body', 'conclusion'}
        if not required_types.issubset(section_types):
            raise ValueError(f'Missing required sections. Must include: {required_types}')
        return v
class LinkedInPostOutput(BaseModel):
    """Pydantic model for LinkedIn post output"""
    content: str = Field(
        description="Main post content",
        max_length=3000
    )
    title: str = Field(
        description="Post headline or title",
        max_length=100
    )
    key_insights: List[str] = Field(
        description="Key professional insights shared in the post",
        min_items=1,
        max_items=5
    )
    image_prompt: str = Field(
        description="Prompt for 1200x627 professional image generation"
    )
    call_to_action: str = Field(
        description="Clear call to action for engagement"
    )
    hashtags: List[str] = Field(
        description="Professional hashtags",
        min_items=1,
        max_items=10
    )
    interactive_elements: Optional[dict] = Field(
        description="Optional poll or other interactive elements",
        default=None
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @validator('content')
    def validate_content_length(cls, v):
        if len(v) > 3000:
            raise ValueError(f'LinkedIn post must not exceed 3000 characters, got {len(v)}')
        return v

    @validator('hashtags')
    def validate_professional_hashtags(cls, v):
        for hashtag in v:
            if not hashtag.startswith('#'):
                raise ValueError(f'Hashtag must start with #: {hashtag}')
            if len(hashtag) > 30:
                raise ValueError(f'Hashtag too long: {hashtag}')
        return v


class ContentGenerator:
    """Class to handle content generation for different platforms"""

    def __init__(
            self,
            model_name: str = "gpt-4",
            temperature: float = 0.3,
            platforms: Optional[List[str]] = None
    ):
        """
        Initialize the content generator

        Args:
            model_name: OpenAI model to use
            temperature: Creativity level
            platforms: List of platforms to initialize (defaults to all)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.platforms = platforms or ["instagram", "linkedin", "blog"]

        # Initialize model
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

        # Initialize parsers and chains
        self.parsers = {}
        self.chains = {}
        self._initialize_chains()

        logging.info(f"ContentGenerator initialized with model: {model_name}")

    def _get_parser(self, platform: str) -> PydanticOutputParser:
        """Get appropriate parser for platform"""
        parser_map = {
            "instagram": InstagramPostOutput,
            "linkedin": LinkedInPostOutput,
            "blog": CompanyBlogOutput
        }

        parser_class = parser_map.get(platform)
        if not parser_class:
            raise ValueError(f"Unsupported platform: {platform}")

        return PydanticOutputParser(pydantic_object=parser_class)

    def _create_prompt_template(self, platform: str) -> ChatPromptTemplate:
        """Create prompt template for platform"""
        return ChatPromptTemplate.from_template(
            """
            {system_prompt}

            {format_instructions}

            Topic or Theme to create content about: {title}

            The Entire Summary of the Text: {content}

            Please generate the content following the format specified above.
            """
        )

    def _initialize_chains(self):
        """Initialize chains for all platforms"""
        for platform in self.platforms:
            try:
                # Get parser
                parser = self._get_parser(platform)
                self.parsers[platform] = parser

                # Create prompt template
                prompt_template = self._create_prompt_template(platform)

                # Create chain
                chain = (
                        prompt_template.partial(
                            system_prompt=PLATFORM_PROMPTS[platform],
                            format_instructions=parser.get_format_instructions()
                        )
                        | self.model
                        | parser
                )

                self.chains[platform] = chain
                logging.info(f"Chain initialized for platform: {platform}")

            except Exception as e:
                logging.error(f"Error initializing chain for {platform}: {str(e)}")
                raise

    def generate_content(
            self,
            content: str,
            title: str,
            platform: Literal["instagram", "linkedin", "blog"]
    ) -> Union[InstagramPostOutput, LinkedInPostOutput, CompanyBlogOutput]:
        """
        Generate content for specified platform

        Args:
            content: Main content text
            title: Content title
            platform: Target platform

        Returns:
            Platform-specific content output
        """
        try:
            if platform not in self.platforms:
                raise ValueError(f"Platform {platform} not initialized")

            # Get chain for platform
            chain = self.chains[platform]

            # Generate content
            result = chain.invoke({
                "content": content,
                "title": title,
            })

            return result

        except Exception as e:
            logging.error(f"Error generating content for {platform}: {str(e)}")
            raise

    def update_model_settings(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """
        Update model settings and reinitialize chains

        Args:
            model_name: New model name (optional)
            temperature: New temperature (optional)
        """
        try:
            if model_name:
                self.model_name = model_name
            if temperature is not None:
                self.temperature = temperature

            # Reinitialize model
            self.model = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature
            )

            # Reinitialize chains
            self._initialize_chains()

            logging.info("Model settings updated and chains reinitialized")

        except Exception as e:
            logging.error(f"Error updating model settings: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    try:
        # Initialize generator once
        generator = ContentGenerator(
            model_name="gpt-4",
            temperature=0.3
        )

        # Generate Instagram content
        instagram_result = generator.generate_content(
            title="Sustainable Living Tips",
            content="Focus on practical, everyday sustainability tips",
            platform="instagram"
        )
        print("\nInstagram Post:")
        print(f"Caption: {instagram_result.caption}")
        print(f"Hashtags: {instagram_result.hashtags}")

        # Generate LinkedIn content
        linkedin_result = generator.generate_content(
            title="Digital Transformation Trends 2024",
            content="Focus on AI and automation impact",
            platform="linkedin"
        )
        print("\nLinkedIn Post:")
        print(f"Title: {linkedin_result.title}")
        print(f"Content: {linkedin_result.content[:100]}...")

        # Generate blog content
        blog_result = generator.generate_content(
            title="The Future of Remote Work",
            content="Include recent statistics and case studies",
            platform="blog"
        )
        print("\nBlog Post:")
        print(f"Title: {blog_result.title}")
        print(f"Word Count: {blog_result.word_count}")

        # Update model settings if needed
        generator.update_model_settings(temperature=0.7)

    except Exception as e:
        print(f"Error: {str(e)}")
