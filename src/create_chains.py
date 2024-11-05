
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Literal, Union, Optional,Any
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseRetriever
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import re

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Base Models
class BlogSection(BaseModel):
    """Model for a blog post section"""
    type: Literal["introduction", "body", "conclusion"]
    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content")
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
        fixed_hashtags = []
        for hashtag in v:
            if not hashtag.startswith('#'):
                fixed_hashtags.append(f"#{hashtag.lstrip('#')}")
            else:
                fixed_hashtags.append(hashtag)
        return fixed_hashtags
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
        fixed_hashtags = []
        for hashtag in v:
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag.lstrip('#')}"
            if len(hashtag) > 30:
                raise ValueError(f'Hashtag too long: {hashtag}')
            fixed_hashtags.append(hashtag)
        return fixed_hashtags
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



# System Prompts
INSTAGRAM_DESCRIPTION = """### Instructions for Instagram Post Creation

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

COMPANY_BLOG_DESCRIPTION = """### Expert Blog Post and Visual Creation for a Company Blog

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
"""

LINKEDIN_DESCRIPTION = """### Instructions for LinkedIn Expert Post Creation

As a LinkedIn expert post writer, your task is to craft an engaging and professional LinkedIn post that adheres to the following guidelines:

1. **Content and Style**: Your post should blend professional insights with a visually appealing format. Focus on sharing authentic stories or showcasing industry expertise. Ensure the tone is professional yet approachable.

2. **Length and Structure**: Aim for a post length of up to 3,000 characters. Structure the post to include a compelling introduction, informative content, and a thought-provoking conclusion or call-to-action.

3. **Visual Elements**: Enhance your post's engagement by incorporating a relevant image. Use dimensions of 1200 x 627 pixels for optimal display.

4. **Engagement Strategies**: Consider integrating video content or interactive elements like polls to boost engagement rates.

5. **Image Generation**: After crafting the post, create a prompt for an image generator API that will produce an image complementing your post content.
"""

# Platform prompts dictionary
PLATFORM_PROMPTS = {
    'linkedin': LINKEDIN_DESCRIPTION,
    'blog': COMPANY_BLOG_DESCRIPTION,
    'instagram': INSTAGRAM_DESCRIPTION
}


class ContentGenerator:
    """Enhanced class to handle content generation with retry logic and retriever"""

    def __init__(
            self,
            model_name: str = "gpt-4",
            temperature: float = 0.3,
            platforms: Optional[List[str]] = None,
            retriever: Optional[BaseRetriever] = None,
            max_retries: int = 3
    ):
        """Initialize content generator with specified settings"""
        self.model_name = model_name
        self.temperature = temperature
        self.platforms = platforms or ["instagram", "linkedin", "blog"]
        self.retriever = retriever
        self.max_retries = max_retries

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
        """Create prompt template with optional retriever context"""
        if self.retriever:
            template = """
            {system_prompt}

            {format_instructions}

            Relevant Context:
            {context}

            Topic or Theme to create content about: {title}

            The Entire Summary of the Text: {content}

            Please generate the content following the format specified above, 
            using the relevant context where appropriate.

            Remember: All output must strictly follow the format specified in the instructions.
            """
        else:
            template = """
            {system_prompt}

            {format_instructions}

            Topic or Theme to create content about: {title}

            The Entire Summary of the Text: {content}

            Please generate the content following the format specified above.

            Remember: All output must strictly follow the format specified in the instructions.
            """

        return ChatPromptTemplate.from_template(template)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ValueError, ValidationError))
    )
    def _generate_with_retry(self, chain, inputs: Dict[str, Any], platform: str):
        """Generate content with retry logic"""
        try:
            result = chain.invoke(inputs)
            self._validate_required_fields(result, platform)
            return result
        except Exception as e:
            logging.warning(f"Generation attempt failed: {str(e)}")
            raise

    def _validate_required_fields(self, result: Any, platform: str):
        """Validate that all required fields are present and properly formatted"""
        required_fields = {
            "instagram": {
                'caption', 'hashtags', 'image_prompt', 'visual_elements',
                'mood', 'engagement_hooks'
            },
            "linkedin": {
                'content', 'title', 'key_insights', 'image_prompt',
                'call_to_action', 'hashtags'
            },
            "blog": {
                'title', 'meta_description', 'keywords', 'sections',
                'word_count', 'image_prompt', 'seo_elements'
            }
        }

        fields = required_fields.get(platform, set())
        missing_fields = fields - set(result.dict().keys())

        if missing_fields:
            raise ValueError(f"Missing required fields for {platform}: {missing_fields}")

    def _initialize_chains(self):
        """Initialize chains for all platforms"""
        for platform in self.platforms:
            try:
                parser = self._get_parser(platform)
                self.parsers[platform] = parser

                prompt_template = self._create_prompt_template(platform)

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

    def _get_retriever_context(self, query: str) -> str:
        """Get relevant context from retriever if available"""
        if not self.retriever:
            return ""

        try:
            relevant_docs = self.retriever.get_relevant_documents(query)
            return "\n\n".join(doc.page_content for doc in relevant_docs)
        except Exception as e:
            logging.error(f"Error retrieving context: {str(e)}")
            return ""

    def generate_content(
            self,
            content: str,
            title: str,
            platform: Literal["instagram", "linkedin", "blog"]
    ) -> Union[InstagramPostOutput, LinkedInPostOutput, CompanyBlogOutput]:
        """Generate content with retry logic and retriever context"""
        try:
            if platform not in self.platforms:
                raise ValueError(f"Platform {platform} not initialized")

            chain = self.chains[platform]

            inputs = {
                "content": content,
                "title": title
            }

            if self.retriever:
                context = self._get_retriever_context(f"{title} {content}")
                inputs["context"] = context

            for attempt in range(self.max_retries):
                try:
                    result = self._generate_with_retry(chain, inputs, platform)
                    return result
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logging.error(f"All retry attempts failed for {platform}")
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed, retrying...")

        except Exception as e:
            logging.error(f"Error generating content for {platform}: {str(e)}")
            raise

    def set_retriever(self, retriever: BaseRetriever):
        """Set or update the retriever"""
        self.retriever = retriever
        self._initialize_chains()


def create_content_batch(
        title: str,
        content: str,
        generator: ContentGenerator
) -> Dict[str, Any]:
    """
    Create content for all platforms in one batch

    Args:
        title: Content title
        content: Main content
        generator: Initialized ContentGenerator instance

    Returns:
        Dict containing results for each platform
    """
    results = {}

    for platform in ["instagram", "linkedin", "blog"]:
        try:
            result = generator.generate_content(
                content=content,
                title=title,
                platform=platform
            )
            results[platform] = result
            logging.info(f"Successfully generated {platform} content")

        except Exception as e:
            logging.error(f"Failed to generate {platform} content: {str(e)}")
            results[platform] = None

    return results


class ContentValidationError(Exception):
    """Custom error for content validation failures"""
    pass


def validate_content_batch(results: Dict[str, Any]) -> bool:
    """
    Validate a batch of generated content

    Args:
        results: Dictionary of generated content

    Returns:
        bool: True if all content is valid

    Raises:
        ContentValidationError: If validation fails
    """
    for platform, content in results.items():
        if content is None:
            raise ContentValidationError(f"Missing content for {platform}")

        if platform == "instagram":
            if not content.hashtags or len(content.hashtags) < 1:
                raise ContentValidationError(f"Invalid hashtags for {platform}")

        elif platform == "linkedin":
            if not content.key_insights or len(content.key_insights) < 1:
                raise ContentValidationError(f"Missing key insights for {platform}")

        elif platform == "blog":
            if not content.sections or len(content.sections) < 3:
                raise ContentValidationError(f"Insufficient sections for {platform}")

    return True


def save_content_batch(
        results: Dict[str, Any],
        output_dir: str = "output"
) -> Dict[str, str]:
    """
    Save generated content to files

    Args:
        results: Dictionary of generated content
        output_dir: Directory to save files

    Returns:
        Dict mapping platforms to file paths
    """
    import os
    import json
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = {}

    for platform, content in results.items():
        if content is not None:
            filename = f"{platform}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content.dict(), f, indent=2, ensure_ascii=False)

            saved_paths[platform] = filepath
            logging.info(f"Saved {platform} content to {filepath}")

    return saved_paths


# Example usage
if __name__ == "__main__":
    try:
        # Initialize generator
        generator = ContentGenerator(
            model_name="gpt-4o",
            temperature=0.3,
            max_retries=3
        )

        # Example content
        sample_title = "The Future of Sustainable Technology"
        sample_content = """
        Sustainable technology is reshaping industries worldwide. From renewable energy 
        to eco-friendly manufacturing, innovations are driving positive environmental change. 
        Companies are increasingly adopting green practices, while consumers demand 
        more sustainable products and services.
        """

        # Optional: Add retriever (implement your own)
        # from langchain.vectorstores import FAISS
        # from langchain.embeddings import OpenAIEmbeddings
        #
        # embeddings = OpenAIEmbeddings()
        # vectorstore = FAISS.from_texts(
        #     ["Your reference text here"],
        #     embeddings
        # )
        # generator.set_retriever(vectorstore.as_retriever())

        # Generate content batch
        print("Generating content...")
        results = create_content_batch(
            title=sample_title,
            content=sample_content,
            generator=generator
        )

        # Validate results
        print("Validating content...")
        try:
            validate_content_batch(results)
            print("Content validation successful!")
        except ContentValidationError as e:
            print(f"Content validation failed: {str(e)}")

        # Save results
        print("Saving content...")
        saved_paths = save_content_batch(results)
        print("Content saved to:", saved_paths)

        # Display results
        for platform, content in results.items():
            if content:
                print(f"\n=== {platform.upper()} CONTENT ===")

                if platform == "instagram":
                    print(f"Caption: {content.caption}")
                    print(f"Hashtags: {' '.join(content.hashtags)}")

                elif platform == "linkedin":
                    print(f"Title: {content.title}")
                    print(f"Content Preview: {content.content[:200]}...")

                else:  # blog
                    print(f"Title: {content.title}")
                    print(f"Word Count: {content.word_count}")
                    print(f"Number of Sections: {len(content.sections)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)


def get_content_statistics(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Generate statistics for the content batch

    Args:
        results: Dictionary of generated content

    Returns:
        Dict containing statistics for each platform
    """
    stats = {}

    for platform, content in results.items():
        if content is None:
            stats[platform] = {"status": "failed"}
            continue

        platform_stats = {"status": "success"}

        if platform == "instagram":
            platform_stats.update({
                "caption_length": len(content.caption),
                "num_hashtags": len(content.hashtags),
                "num_visual_elements": len(content.visual_elements)
            })

        elif platform == "linkedin":
            platform_stats.update({
                "content_length": len(content.content),
                "num_insights": len(content.key_insights),
                "num_hashtags": len(content.hashtags)
            })

        elif platform == "blog":
            platform_stats.update({
                "word_count": content.word_count,
                "num_sections": len(content.sections),
                "num_keywords": len(content.keywords)
            })

        stats[platform] = platform_stats

    return stats


