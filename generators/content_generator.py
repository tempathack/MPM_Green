import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Literal, Tuple
from openai import OpenAI
import langdetect
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Change relative imports to absolute
from generators.image_generator import ImageGenerator
from models.output_models import (
    InstagramPostOutput,
    LinkedInPostOutput,
    CompanyBlogOutput
)
from prompts.system_prompts import PLATFORM_PROMPTS


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

        # Initialize language detection
        langdetect.DetectorFactory.seed = 0

        # Initialize OpenAI client
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
        self.initialize_chains()

        logging.info(f"ContentGenerator initialized with model: {model_name}")

    def detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            return detect(text)
        except LangDetectException as e:
            logging.warning(f"Language detection failed: {str(e)}. Defaulting to 'en'")
            return 'en'
        except Exception as e:
            logging.error(f"Unexpected error in language detection: {str(e)}")
            return 'en'

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
        """Create prompt template"""
        template = """
        {system_prompt}

        {format_instructions}

        Topic or Theme: {title}
        Content: {content}
        Language: {language}
        Context: {context}

        Please generate content following the format above.
        """
        return ChatPromptTemplate.from_template(template)

    def initialize_chains(self):
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ValueError, ValidationError))
    )
    async def _generate_with_retry(self, chain, inputs: Dict[str, Any], platform: str):
        """Generate content with retry logic"""
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: chain.invoke(inputs)
            )

            if platform == "blog" and hasattr(result, 'word_count') and result.word_count < 600:
                result = await self._enhance_blog_content(result)

            return result
        except Exception as e:
            logging.warning(f"Generation attempt failed: {str(e)}")
            raise

    async def _translate_prompt(
            self,
            prompt: str,
            source_lang: str,
            target_lang: str = 'en'
    ) -> str:
        """Translate prompt to target language"""
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": f"Translate from {source_lang} to {target_lang}, maintaining descriptive quality."
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

    async def get_relevant_context(self, query: str) -> str:
        """Get relevant context from retriever"""
        if not self.retriever:
            return ""

        try:
            docs = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.retriever.get_relevant_documents(query)
            )
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logging.error(f"Error retrieving context: {str(e)}")
            return ""

    async def generate_content_with_image(
            self,
            content: str,
            title: str,
            platform: Literal["instagram", "linkedin", "blog"]
    ) -> Union[InstagramPostOutput, LinkedInPostOutput, CompanyBlogOutput]:
        """Generate content and image"""
        try:
            if platform not in self.platforms:
                raise ValueError(f"Platform {platform} not initialized")

            # Get chain and detect language
            chain = self.chains[platform]
            lang = self.detect_language(f"{title} {content}")

            # Get context if available
            context = await self.get_relevant_context(f"{title} {content}")

            # Prepare inputs
            inputs = {
                "content": content,
                "title": title,
                "language": lang,
                "context": context or ""  # Ensure empty string if no context
            }

            # Generate content
            result = await self._generate_with_retry(chain, inputs, platform)

            # Generate image if available
            if result and hasattr(result, 'image_prompt'):
                # Handle translation if needed
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

                if image_url:
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

    async def _enhance_blog_content(self, blog_content: CompanyBlogOutput) -> CompanyBlogOutput:
        """Enhance blog content to meet minimum word count"""
        try:
            enhancement_prompt = f"""
            Please expand this blog content to at least 600 words while maintaining quality.
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