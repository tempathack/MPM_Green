from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Literal, Union, Optional, Any, Tuple
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseRetriever
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import re
import asyncio
from openai import OpenAI
import langdetect
import aiohttp
from pathlib import Path

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO)


class ImageGenerator:
    """Handle image generation with OpenAI's DALL-E"""

    def __init__(self, client: OpenAI):
        self.client = client

    async def generate_image(self, prompt: str, size: str = "1024x1024") -> Tuple[str, str]:
        """
        Generate image from prompt

        Args:
            prompt: Image generation prompt
            size: Image size (1024x1024 or 1792x1024)

        Returns:
            Tuple[str, str]: Image URL and revised prompt
        """
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size=size,
                    quality="standard",
                    n=1
                )
            )

            return response.data[0].url, response.data[0].revised_prompt

        except Exception as e:
            logging.error(f"Error generating image: {str(e)}")
            return None, prompt


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
    generated_image: Optional[Dict[str, str]] = Field(
        description="Generated image data including URL and prompts",
        default=None
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
    generated_image: Optional[Dict[str, str]] = Field(
        description="Generated image data including URL and prompts",
        default=None
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
    generated_image: Optional[Dict[str, str]] = Field(
        description="Generated image data including URL and prompts",
        default=None
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
"""

LINKEDIN_DESCRIPTION = """### Instructions for LinkedIn Expert Post Creation

As a LinkedIn expert post writer, your task is to craft an engaging and professional LinkedIn post that adheres to the following guidelines:

1. **Content and Style**: Your post should blend professional insights with a visually appealing format. Focus on sharing authentic stories or showcasing industry expertise. Ensure the tone is professional yet approachable.

2. **Length and Structure**: Aim for a post length of up to 3,000 characters. Structure the post to include a compelling introduction, informative content, and a thought-provoking conclusion or call-to-action.

3. **Visual Elements**: Enhance your post's engagement by incorporating a relevant image. Use dimensions of 1200 x 627 pixels for optimal display.

4. **Engagement Strategies**: Consider integrating video content or interactive elements like polls to boost engagement rates.

5. **Image Generation**: After crafting the post, create a prompt for an image generator API that will produce an image complementing your post content.
"""

PLATFORM_PROMPTS = {
    'linkedin': LINKEDIN_DESCRIPTION,
    'blog': COMPANY_BLOG_DESCRIPTION,
    'instagram': INSTAGRAM_DESCRIPTION
}

class ContentGenerator:
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

        # Initialize OpenAI client for both text and image generation
        self.openai_client = OpenAI()
        self.image_generator = ImageGenerator(self.openai_client)

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

    def detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            return langdetect.detect(text)
        except:
            return 'en'

    async def _translate_prompt(
        self,
        prompt: str,
        source_lang: str,
        target_lang: str = 'en'
    ) -> str:
        """Translate prompt to English for image generation"""
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": f"Translate the following text from {source_lang} to {target_lang}, maintaining the descriptive quality needed for image generation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return prompt

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
        """Create prompt template with enhanced instructions for content length"""
        if platform == "blog":
            template = """
            {system_prompt}

            {format_instructions}

            Topic or Theme to create content about: {title}

            The Entire Summary of the Text: {content}

            IMPORTANT REQUIREMENTS:
            1. Meta description MUST be under 160 characters
            2. Total word count MUST be between 600-2000 words
            3. Each section must be detailed and comprehensive
            4. All sections (introduction, body, conclusion) are required

            Please generate the content following the format specified above.
            Language detected: {language}
            """
        elif platform == "instagram":
            template = """
            {system_prompt}

            {format_instructions}

            Topic or Theme to create content about: {title}

            The Entire Summary of the Text: {content}

            IMPORTANT REQUIREMENTS:
            1. Caption must be under 125 characters
            2. Include at least 3 relevant hashtags
            3. Hashtags must start with #
            4. Image prompt must be detailed and specific

            Please generate the content following the format specified above.
            Language detected: {language}
            """
        else:  # linkedin
            template = """
            {system_prompt}

            {format_instructions}

            Topic or Theme to create content about: {title}

            The Entire Summary of the Text: {content}

            IMPORTANT REQUIREMENTS:
            1. Content must be under 3000 characters
            2. Include at least 3 key insights
            3. Hashtags must start with #
            4. Include a clear call to action

            Please generate the content following the format specified above.
            Language detected: {language}
            """
        return ChatPromptTemplate.from_template(template)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ValueError, ValidationError))
    )
    async def _generate_with_retry(self, chain, inputs: Dict[str, Any], platform: str):
        """Generate content with retry logic and content enhancement"""
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: chain.invoke(inputs)
            )

            # Enhance content if needed
            if platform == "blog" and result.word_count < 600:
                result = await self._enhance_blog_content(result)

            await self._validate_required_fields(result, platform)
            return result
        except Exception as e:
            logging.warning(f"Generation attempt failed: {str(e)}")
            raise

    async def _enhance_blog_content(self, blog_content: CompanyBlogOutput) -> CompanyBlogOutput:
        """Enhance blog content to meet minimum word count"""
        try:
            enhancement_prompt = f"""
            Please expand the following blog content to be at least 600 words while maintaining quality and relevance.
            Current word count: {blog_content.word_count}

            Original content:
            {blog_content.dict()}
            """

            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert content enhancer."},
                        {"role": "user", "content": enhancement_prompt}
                    ]
                )
            )

            enhanced_content = response.choices[0].message.content
            parser = self._get_parser("blog")
            return parser.parse_raw(enhanced_content)

        except Exception as e:
            logging.error(f"Error enhancing blog content: {str(e)}")
            raise



    async def _validate_required_fields(self, result: Any, platform: str):
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
    async def validate_content_batch(results: Dict[str, Any]) -> bool:
        """Validate content batch with enhanced error handling"""
        validation_errors = []

        for platform, content in results.items():
            try:
                if content is None:
                    validation_errors.append(f"Missing content for {platform}")
                    continue

                if platform == "instagram":
                    if not content.hashtags or len(content.hashtags) < 1:
                        validation_errors.append(f"Invalid hashtags for {platform}")
                    if len(content.caption) > 125:
                        validation_errors.append(f"Caption too long for {platform}")

                elif platform == "linkedin":
                    if not content.key_insights or len(content.key_insights) < 1:
                        validation_errors.append(f"Missing key insights for {platform}")
                    if len(content.content) > 3000:
                        validation_errors.append(f"Content too long for {platform}")

                elif platform == "blog":
                    if not content.sections or len(content.sections) < 3:
                        validation_errors.append(f"Insufficient sections for {platform}")
                    if content.word_count < 600 or content.word_count > 2000:
                        validation_errors.append(
                            f"Invalid word count for {platform}: {content.word_count}"
                        )

                # Validate image generation
                if not content.generated_image or not content.generated_image.get('url'):
                    logging.warning(f"Missing generated image for {platform}")

            except Exception as e:
                validation_errors.append(f"Error validating {platform}: {str(e)}")

        if validation_errors:
            raise ContentValidationError("\n".join(validation_errors))

        return True

    async def generate_content_with_image(
            self,
            content: str,
            title: str,
            platform: Literal["instagram", "linkedin", "blog"]
    ) -> Union[InstagramPostOutput, LinkedInPostOutput, CompanyBlogOutput]:
        """Generate content and image asynchronously"""
        try:
            if platform not in self.platforms:
                raise ValueError(f"Platform {platform} not initialized")

            chain = self.chains[platform]

            # Detect language
            lang = self.detect_language(content)

            # Prepare inputs
            inputs = {
                "content": content,
                "title": title,
                "language": lang
            }

            # Generate content
            result = await self._generate_with_retry(chain, inputs, platform)

            # Generate image if content generation succeeded
            if result and hasattr(result, 'image_prompt'):
                # Translate image prompt if not in English
                if lang != 'en':
                    translated_prompt = await self._translate_prompt(
                        result.image_prompt,
                        source_lang=lang,
                        target_lang='en'
                    )
                else:
                    translated_prompt = result.image_prompt

                # Generate image
                image_url, revised_prompt = await self.image_generator.generate_image(
                    translated_prompt
                )

                # Update result with image data
                result_dict = result.dict()
                result_dict['generated_image'] = {
                    'url': image_url,
                    'revised_prompt': revised_prompt,
                    'original_prompt': result.image_prompt
                }

                # Convert back to appropriate model
                parser = self._get_parser(platform)
                return parser.parse_obj(result_dict)

            return result

        except Exception as e:
            logging.error(f"Error generating content with image: {str(e)}")
            raise


async def create_content_batch(
        title: str,
        content: str,
        generator: ContentGenerator
) -> Dict[str, Any]:
    """Create content for all platforms in one batch asynchronously"""
    results = {}

    async def generate_for_platform(platform: str):
        try:
            result = await generator.generate_content_with_image(
                content=content,
                title=title,
                platform=platform
            )
            return platform, result
        except Exception as e:
            logging.error(f"Failed to generate {platform} content: {str(e)}")
            return platform, None

    # Generate content for all platforms concurrently
    tasks = [generate_for_platform(platform) for platform in generator.platforms]
    completed_tasks = await asyncio.gather(*tasks)

    # Collect results
    for platform, result in completed_tasks:
        results[platform] = result

    return results


class ContentValidationError(Exception):
    """Custom error for content validation failures"""
    pass


async def validate_content_batch(results: Dict[str, Any]) -> bool:
    """Validate a batch of generated content"""
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

        # Validate image generation
        if not content.generated_image or not content.generated_image.get('url'):
            logging.warning(f"Missing generated image for {platform}")

    return True


async def save_content_batch(
        results: Dict[str, Any],
        output_dir: str = "output"
) -> Dict[str, str]:
    """Save generated content to files"""
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

            # Save image if available
            if content.generated_image and content.generated_image.get('url'):
                try:
                    image_path = os.path.join(output_dir, f"{platform}_{timestamp}.png")
                    async with aiohttp.ClientSession() as session:
                        async with session.get(content.generated_image['url']) as response:
                            if response.status == 200:
                                image_data = await response.read()
                                with open(image_path, 'wb') as f:
                                    f.write(image_data)
                                logging.info(f"Saved image for {platform} to {image_path}")
                except Exception as e:
                    logging.error(f"Failed to save image for {platform}: {str(e)}")

    return saved_paths


# Example usage
async def main():
    try:
        # Initialize generator
        generator = ContentGenerator(
            model_name="gpt-4",
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

        # Generate content batch
        print("Generating content...")
        results = await create_content_batch(
            title=sample_title,
            content=sample_content,
            generator=generator
        )

        # Validate results
        print("Validating content...")
        try:
            await validate_content_batch(results)
            print("Content validation successful!")
        except ContentValidationError as e:
            print(f"Content validation failed: {str(e)}")

        # Save results
        print("Saving content...")
        saved_paths = await save_content_batch(results)
        print("Content saved to:", saved_paths)

        # Display results
        for platform, content in results.items():
            if content:
                print(f"\n=== {platform.upper()} CONTENT ===")
                if platform == "instagram":
                    print(f"Caption: {content.caption}")
                    print(f"Hashtags: {' '.join(content.hashtags)}")
                    if content.generated_image:
                        print(f"Image URL: {content.generated_image['url']}")
                elif platform == "linkedin":
                    print(f"Title: {content.title}")
                    print(f"Content Preview: {content.content[:200]}...")
                    if content.generated_image:
                        print(f"Image URL: {content.generated_image['url']}")
                else:  # blog
                    print(f"Title: {content.title}")
                    print(f"Word Count: {content.word_count}")
                    print(f"Number of Sections: {len(content.sections)}")
                    if content.generated_image:
                        print(f"Image URL: {content.generated_image['url']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())