import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Literal, Tuple,Type

from dotenv import load_dotenv
from openai import OpenAI
import langdetect
import asyncio
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models.output_models import (
    InstagramPostOutput,
    LinkedInPostOutput,
    CompanyBlogOutput,BaseContentOutput
)
from prompts.system_prompts import PLATFORM_PROMPTS


class ContentGenerator:
    def __init__(
            self,
            model_name: str = "gpt-4o",
            temperature: float = 0.3,
            platforms: Optional[List[str]] = None,
            corpus:str=None,
            max_retries: int = 10,
            lang:str='de'

    ):
        """Initialize content generator with specified settings"""
        self.model_name = model_name
        self.lang=lang
        self.temperature = temperature
        self.platforms = platforms or ["instagram", "linkedin", "blog"]
        self.corpus=corpus
        self.max_retries = max_retries
        self.model_map: Dict[str, Type[BaseContentOutput]] = {
            "image_post": InstagramPostOutput,
            "linkedin_post": LinkedInPostOutput,
            "blog_post": CompanyBlogOutput,
        }

        # Initialize language detection
        langdetect.DetectorFactory.seed = 0

        # Initialize OpenAI client
        self.openai_client = OpenAI()

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
            "instagram":InstagramPostOutput,
            "linkedin":LinkedInPostOutput,
            "blog": CompanyBlogOutput,
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
                    model= self.model_name,
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

    async def parse_content_output(self,data: dict) -> BaseContentOutput:
        content_type = data.get("content_type")


        model_cls = self.model_map.get(content_type)
        if model_cls is None:
            raise ValueError(f"Unknown content type: {content_type}")

        return model_cls.parse_obj(data)
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



            # Prepare inputs
            inputs = {
                "content": content,
                "title": title,
                "language": self.lang,
            }

            # Generate content
            result = await self._generate_with_retry(chain, inputs, platform)






            result_dict = result.dict()



            return self.parse_content_output(result_dict)


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
                    model="gpt-4o",
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


async def generate_content_for_all_platforms(contents: List[str], titles: List[str],images:List[Any], **kwargs) -> List[Dict]:
    """
    Generates content for each platform using given lists of contents and titles.

    Parameters:
        contents (List[str]): List of content texts.
        titles (List[str]): List of titles corresponding to each content.
        kwargs: Additional arguments for ContentGenerator, like `model_name`, `max_retries`, `temperature`.

    Returns:
        List[Dict]: List of results for each content-title-platform combination.
    """
    # Pass **kwargs to ContentGenerator to initialize with flexible parameters
    generator = ContentGenerator(**kwargs)



    res = []
    for content, title in zip(contents, titles):
        for platform in generator.platforms:
            # Generate content with image for each content-title-platform combination
            result = await generator.generate_content_with_image(content=content, title=title, platform=platform)
            obtained = await result  # Await the result for each platform
            res.append({
                "platform": platform,
                "title": title,
                "content": content,
                "result": obtained,
                'images':images
            })

    return res


if __name__ == '__main__':
    load_dotenv()
    content_list = [
        """
        Many industries pay employees well above average for specialized skills or senior roles. 
        "Overpaid" is a term that describes cases where compensation is seen as disproportionately 
        high compared to the work done or the market rate. Critics argue that such salaries contribute 
        to economic inequality, while supporters believe they reward talent and foster innovation.
        """,
        """
        Sustainability is becoming a top priority for businesses worldwide. Adopting eco-friendly 
        practices not only benefits the environment but also builds a positive brand image. 
        Companies are increasingly exploring ways to reduce carbon footprints, from sustainable 
        sourcing to waste reduction initiatives.
        """
    ]

    title_list = [
        "The Concept of Overpaid Employees",
        "Sustainable Business Practices: A New Era of Responsibility"
    ]

    res=asyncio.run(generate_content_for_all_platforms(content_list, title_list, model_name="gpt-4o", max_retries=10, temperature=0.3))
    print(res)